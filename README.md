# live-listen

Ecoute temps reel + transcription + resume progressif. Pense pour suivre un
film, un cours ou une conference : le son est capture par chunks
(`pw-record` 16 kHz mono, direct whisper-optimal, wavs temporaires supprimes
en fin de session), filtre par une porte d'energie, transcrit avec whisper.cpp
en local, puis un resume progressif unique est maintenu via le serveur OpenVINO
local.

Pipeline : `pw-record` (sur thread dedie) -> porte RMS -> `whisper-cli` ->
`transcription.txt` -> serveur OpenVINO (`resume.txt`). 100 % local, stdlib
Python uniquement (`urllib`, pas de `requests`), zero dependance pip.

La capture ne s'interrompt jamais pendant la transcription : un thread
d'enregistrement dedie produit les chunks pendant qu'un second thread les
transcrit, donc la cadence effective est `chunk` et non `chunk + latence STT`.

## Install

```text
cd ~/Labs/KpihX-Labs/AI/live-listen
uv tool install -e .     # ou : make install (enchaine avec live-listen init)
live-listen init         # cree ~/.local/share/live-listen/config.json
```

La config est creee dans `~/.local/share/live-listen/` (XDG data dir,
jamais dans le projet). `make install` enchaine install outil + init.
Sans install, la config est creee au premier lancement (message
`[config] cree : <chemin>`).

## Requis

Binaires : `pw-record`, `wpctl`, `ffmpeg` (repli enregistrement),
`whisper-cli`, serveur OpenVINO sur `http://127.0.0.1:11437`
(`openvino ps` pour verifier, `openvino server start` sinon).
Pas de `pactl`, pas de lib audio Python : tout passe par des sous-processes.
Un binaire absent ne casse plus la session : le pretraitement le signale et
lance refuse, au lieu d'une traceback qui tue la capture.

Modeles whisper dans `~/.local/share/whisper/models/ggml/`
(`ggml-large-v3-turbo-q5_0.bin` par defaut).

## Pretraitement

Active par defaut (`preflight: true`), avant le premier chunk :

1. `pw-record` et `whisper-cli` presents sur le PATH.
2. Le modele whisper configure existe.
3. **Coherence sink/monitor** : compare le `monitor_target` au sink audio par
   defaut. C'est le piege le plus coutant en usage reel : un monitor qui
   appartient au haut-parleur interne alors que le son sort par un casque
   Bluetooth enregistre un silence parfait pendant toute la session. Un
   avertissement explicite est emis, la decision reste a l'utilisateur.
4. Serveur OpenVINO joignable **et** `summary_model` effectivement servi.

`--no-preflight` lance quand meme (diagnostic).

## Porte d'energie et hallucinations

Whisper invente du texte quand il n'y a rien a transcrire. Deux pieges distincts,
deux garde-fous distincts, et il ne faut pas les confondre.

**1. Entree morte ou mauvaise source.** Un micro coupe, ou un monitor qui
appartient a un sink par lequel le son ne passe pas (le piege Bluetooth), ne
produisent que du bruit de fond. `silence_rms` (defaut `0.002`, RMS normalise
0..1) ecarte ces chunks avant whisper. `0` desactive la porte.

Le seuil est **calibre sur les chunks reels du 2026-09-23**, pas choisis au
hasard : le chunk qui portait la vraie parole test ("Quand on est a
l'interieur, on enleve son echarpe") mesure RMS 0.0079. Un seuil proche de
0.008 jetterait cette parole ; d'ou 0.002, sous la parole la plus douce mesuree
et au-dessus du bruit de fond. Sur les 45 chunks archives, les 45 passent.

**2. Jingle ou musique, pas du silence.** `Sous-titrage Societe
Radio-Canada` repete 7 fois etait un vrai signal audio, avec une vraie energie :
aucune porte d'energie ne peut le distinguer de la parole. C'est le role du
repli des repetitions consecutives (`Radio-Canada x3` : 9 mots -> 3). Limite
assumee : le repli est exact, donc une boucle qui derive en se repetant n'est
que raccourcie, pas supprimee. Le `max_new_chars` borne de toute facon la
croissance.

**Le VAD natif de whisper-cli.** `whisper-cli` embarque lui-meme une Voice
Activity Detection neurale (Silero) : `--vad`, `--vad-model`,
`--vad-threshold`. C'est un vrai separateur parole/non-parole, bien meilleur
qu'une porte d'energie. Il n'est **pas actif par defaut** parce qu'il exige un
modele VAD au format GGML qui n'est pas installe ici.

Les drapeaux ne sont emis que si le fichier existe vraiment : envoyer `--vad`
sans `--vad-model` ferait echouer whisper-cli sur chaque chunk. Une fois le
modele depose dans `~/.local/share/whisper/models/vad/`, il suffit de :

```
live-listen --vad-model ~/.local/share/whisper/models/vad/<modele-vad-ggml>.bin --vad-threshold 0.5
```

et c'est actif, sans toucher au code. `vad_model: null` par defaut, donc le
comportement actuel est inchange. Le pretraitement dit explicitement dans quel
etat il est.

## Modele de resume : le raisonnement et les latences

`enable_thinking=false` **n'est pas autoritaire** sur le checkpoint Qwen3
sert ici : le modele peut encore s'auto-emettre un bloc `<think>` malgre le
bloc de template vide. Mesure deux fois le 2026-09-23, un resume ecrit en
anglais ("Okay, let's start by looking at the previous summary...") a la
place du resume francais.

`think_directive: true` (defaut) prefixe `/no_think` au prompt systeme. C'est
la directive de template qui survit a `enable_thinking=false`. Mettre
`think_directive: false` pour une famille qui ne connait pas ce jeton
(phi, gemma, llama) : la config emet alors un avertissement.

`top_p` (0.95) et `top_k` (64) sont toujours envoyes. Les defauts OpenVINO
sont `top_p=1.0` et `top_k=2**64-1`, c'est-a-dire **aucune coupe** : avec une
temperature non nulle un modele de classe Gemma echantillonne tout le
vocabulaire et repond en bruit multilingue (mesure : 2 min 20 s de bruit a
temperature 0.7, contre 1,78 s de reponse coherente avec cette coupe).

En plus de `/no_think`, le filtre de raisonnement est pondere et ancre sur des
signaux de structure, avec deux fixtures de regression qui sont les deux fuites
reelles du 2026-09-23 (pas des textes reecrits). Le resume n'est ecrit que si
le score passe sous le seuil, sinon un seul retry avec une consigne plus
brutale, sinon l'ancien resume est conserve.

## Modeles disponibles

Le serveur OpenVINO local sert : `qwen3-0.6b`, `qwen3-1.7b`,
`qwen3-vl-2b-instruct`, `gemma-4-12b-it`. Les deux Qwen3 sont des modeles a
raisonnement, d'ou `/no_think`.

Candidats **sans raisonnement**, pour un resume vraiment fluide, pulls
possibles avec `openvino pull <id>` :

| Modele | Taille | Reflexion | Remarque |
|--------|--------|-----------|----------|
| `microsoft/Phi-4-mini-instruct` | 2.2G | non, par construction | chez Microsoft le raisonnement est un modele **separe** (`Phi-4-mini-reasoning`) |
| `OpenVINO/Phi-4-mini-instruct-int4-ov` | 2.2G | non | IR Intel deja publiee, INT4 NNCF |
| `OpenVINO/Phi-3.5-mini-instruct-int4-ov` | 1.9G | non | IR Intel publiee |
| `Qwen3-4B-Instruct-2507` | ~2.1G | non, `<think>` impossible | le correctif structurel : plus besoin de filtrer |

`Phi-4-mini-instruct` repond a la question "un modele rapide qui ne raisonne
pas" : c'est exactement le pendant Microsoft de `Phi-4-mini-reasoning`, donc
aucun bloc `<think>` n'est a filtrer. `Qwen3-4B-Instruct-2507` est l'autre
voie : un Qwen3 dont la carte le declare incapable de produire un `<think>`,
ce qui supprime la cause au lieu de la filtrer. Attention, les deux sont bien
plus gros que `qwen3-0.6b` : la taille n'est pas la|latence, mesurer avant de
choisir.

**Si tu utilises Phi ou Gemma, mets `think_directive: false`.** Le jeton
`/no_think` n'a de sens que pour la famille Qwen ; ailleurs il serait lu comme
du texte ordinaire. La config detecte le cas et avertit au demarrage :

```
live-listen --summary-model phi-4-mini-instruct --no-think-directive
```

## Mesures sur cette machine (2026-09-30, meme transcription test)

Sortie propre (score 0 au detecteur) dans tous les cas. Le temps inclut le
chargement du modele au premier appel, donc seul l'ordre de grandeur compte.

| Modele | Temps | Mots | Fidelite constatee |
|--------|-------|------|--------------------|
| `qwen3-0.6b` + `/no_think` | 10.8 s (tiede) | ~110 | colle a la transcription |
| `qwen3-1.7b` + `/no_think` | 15.9 s | 114 | colle a la transcription, meilleur francais |
| `gemma-4-12b-it` | 52.0 s | 82 | propre mais 3x plus lent |
| `phi-4-mini-instruct` | 39.1 s (1er appel, chargement inclus) | 99 | **invente** : "se deshabille", "culture hip-hop", absents de la transcription |

Recommandation mesuree : `qwen3-1.7b` + `/no_think` par defaut (le plus fidele
pour 16 s), `qwen3-0.6b` si la vitesse prime. Phi-4-mini ne gagne ni en
vitesse ni en fidelite sur cet echantillon ; a re-mesurer a chaud avant tout
changement de defaut.

## Un piege GPU a connaitre

Mesure sur cette machine : charger un second modele alors que le premier est
encore resident **fait crasher le worker OpenVINO**. Le log serveur montre
`clFinish, error code: -5 CL_OUT_OF_RESOURCES` puis
`terminate called after throwing an instance of 'ov::Exception'`, et toutes les
requetes suivantes repondent 429 ou 502.

Concretement : **un seul modele a la fois**. Ne change pas `summary_model` en
cours de session, et ne lance pas un autre client OpenVINO pendant que
live-listen tourne. live-listen ne fait qu'une requete a la fois (un seul
thread de resume), il n'est donc pas lui la cause, mais il nomme maintenant
l'erreur au lieu d'afficher un HTTP opaque.

## Robustesse

| Symptome | Traitement |
|---|---|
| Binaire absent | code 127/126, erreur claire, pas de traceback |
| Boucle d'enregistrement qui tourne en boucle serree | backoff `record_backoff`, abandon apres `record_max_failures` |
| Serveur injoignable | ancien resume conserve, cadence du retry qui backs-off, prompt borne par `max_new_chars` |
| `Ctrl-C` pendant l'ecriture du resume | ecriture atomique (fichier temp + `fsync` + rename), jamais de `resume.txt` tronque |
| Transcription plus lente que le temps reel | chunks abandonnes et comptes, jamais de blocage du micro |
| Reponse vide du serveur | erreur nommee, y compris le cas `reasoning_content` seul |
| 429 / 502 du serveur | nommes comme crash probable du worker GPU (`CL_OUT_OF_RESOURCES`), avec la commande de diagnostic |
| Interleaving des logs | `log()` serialise par un verrou |

Le resume est un **fichier unique ecrase** a chaque mise a jour (un seul resume
progressif, pas un journal) : les mises a jour precedentes ne sont pas conservees.


## Config

`~/.local/share/live-listen/config.json` (XDG data dir, cree par
`live-listen init` ou au premier lancement) contient TOUS les defauts
(langue, consigne, modeles, chunk, cadence des resumes, source, monitor,
serveur...). Note : l'ancien `./config.json` a la racine du projet n'est
plus lu (decision XDG), il a ete supprime.
Precedence : flags CLI > `--config FICHIER` > config XDG > repli
integre. Chaque cle de config a un flag CLI de meme nom.
`--prompt-file` est l'alternative a `--prompt` (le fichier gagne si les
deux sont donnes). La config effective est affichee dans la console au
demarrage (lignes `[config]`), aucun fichier `config.effective.json`.

## Reference des cles de config

Chaque cle a un flag CLI de meme nom (`--max-new-chars` pour `max_new_chars`).
Une cle absente du fichier garde sa valeur par defaut ; le fichier est cree au
premier lancement par `live-listen init`.

| Cle | Defaut | Role |
|-----|--------|------|
| `lang` | `es` | langue de transcription whisper |
| `prompt` / `prompt_file` | (consigne FR) | consigne de resume ; le fichier gagne |
| `stt_model` | `ggml-large-v3-turbo-q5_0.bin` | modele whisper |
| `stt_device` | `auto` | prefere XPU, puis CUDA, puis GPU, puis CPU ; choix explicite possible |
| `stt_timeout` | `600` | delai max d'un appel whisper |
| `chunk` | `15` | duree d'un chunk, secondes |
| `summary_every` | `90` | cadence des resumes, secondes |
| `summary_model` | `qwen3-1.7b` | modele de resume servi par OpenVINO |
| `thinking` | `false` | `enable_thinking` dans le corps de la requete |
| `think_directive` | `true` | prefixe `/no_think` (Qwen3 uniquement) |
| `top_p` / `top_k` | `0.95` / `64` | coupe d'echantillonnage (les defauts OpenVINO ne coupent rien) |
| `temperature` | `0.0` | 0 = glouton, donc reproductible |
| `max_tokens` | `2048` | budget de la reponse |
| `max_new_chars` | `4000` | plafond de transcription envoyee par resume |
| `http_timeout` | `180` | delai max d'un appel resume |
| `silence_rms` | `0.002` | porte d'energie avant STT ; `0` desactive |
| `vad_model` / `vad_threshold` | `null` / `0.5` | VAD natif whisper-cli, actif si le modele existe |
| `overlap` | `true` | enregistrement pendant la transcription |
| `record_backoff` | `5` | pause apres un echec d'enregistrement |
| `record_max_failures` | `10` | echecs consecutifs avant abandon |
| `preflight` | `true` | verifications avant le premier chunk |
| `source` | `monitor` | `monitor` / `mic` / `auto` |
| `monitor_target` | (haut-parleurs internes) | noeud PipeWire capture |
| `server` | `http://127.0.0.1:11437` | serveur OpenVINO |
| `transcript_path` / `summary_path` | `null` | voir la regle des 4 cas |


## Exemples

Film en espagnol via les haut-parleurs, consigne cinema (cas nominal) :

```
live-listen --lang es --chunk 15 --summary-every 90 --prompt "Tu es l'assistant d'un cours d'espagnol consacre au cinema. Maintiens UN resume progressif unique : intrigue, personnages, vocabulaire notable, tournures idiomatiques, references culturelles. Reponds en francais."
```

Cours ou amphi via le micro :

```
live-listen --lang fr --source mic --summary-model qwen3-0.6b
```

Autre langue, modele plus gros, thinking active :

```
live-listen --lang it --summary-model qwen3-1.7b --thinking
```

Chunks courts et resumes frequents (suivi serre) :

```
live-listen --lang es --chunk 10 --summary-every 30
```

Lister les sources sans enregistrer (sonde le monitor 2 s) :

```
live-listen --list-sources
```

Config alternative (ex. session anglaise, STT leger `ggml-small.en.bin`,
fourni avec le projet) :

```
live-listen --config configs/en-small.json --lang en
```

Consigne de resume depuis un fichier (prioritaire sur `--prompt`) :

```
live-listen --lang es --prompt-file prompt-cinema.txt
```

## Sorties (pas de sessions)

Pas de `--session-dir`, pas de dossier session, pas de `chunks/` archive :
les wavs des chunks vont dans un repertoire temporaire (supprime a la
fin). Les deux `.txt` sont les seules sorties. Les anciens dossiers
`sessions/` restent en archives, l'outil n'y touche plus.

Le repertoire de travail courant au lancement recoit (selon la regle
en 4 cas ci-dessous) :

- `<transcription>.txt` : segments horodates `[HH:MM:SS] texte`
- `<resume>.txt` (a cote ou ailleurs, voir section ci-dessous) : resume
  progressif (`Mise a jour #N`, horodate, modele)

## Arret

`Ctrl-C` : arret propre, fichiers conserves, chemins affiches.
Un resume en cours a jusqu'a `http_timeout + 30 s` pour se terminer avant la
sortie. Passe ce delai il est abandonne, ce qui est sans risque : le fichier
resume est ecrit de facon atomique, donc il reste dans son etat precedent
plutot que d'etre tronque.

L'arret de l'enregistreur peut prendre jusqu'a `chunk + 20 s` s'il est en
cours d'enregistrement.

## Chemins de sortie : `--transcript-path` / `--summary-path`

Deux sorties `.txt` uniquement : le fichier transcription (segments
horodates `[HH:MM:SS] texte`) et le fichier resume (resume progressif en
texte brut, memes lignes d'en-tete : numero de mise a jour, horodate,
modele). Plus de miroirs `.md`, plus de sidecars.
Dans la config, les defauts sont `"transcript_path": null` et
`"summary_path": null` (`null` = non fourni). Precedence inchangee :
flags CLI > `--config FICHIER` > config XDG > repli integre.

Regle de resolution (exactement 4 cas) :

1. Ni l'un ni l'autre -> les deux dans le repertoire courant :
   `transcription.txt` + `resume.txt`.
   Ex. : `live-listen --lang es`
2. Seul `--transcript-path` donne -> le resume (`resume.txt`) est cree
   A COTE, meme dossier que le transcript donne.
   Ex. : `live-listen --transcript-path /tmp/t1/X.txt`
   -> `/tmp/t1/X.txt` + `/tmp/t1/resume.txt`
3. Seul `--summary-path` donne -> la transcription (`transcription.txt`)
   est creee A COTE, meme dossier que le resume donne.
   Ex. : `live-listen --summary-path /tmp/t2/Y.txt`
   -> `/tmp/t2/transcription.txt` + `/tmp/t2/Y.txt`
4. Les deux donnes -> respectes verbatim.
   Ex. : `live-listen --transcript-path /tmp/t3/T.txt --summary-path /tmp/t4/S.txt`
   -> `/tmp/t3/T.txt` + `/tmp/t4/S.txt`
