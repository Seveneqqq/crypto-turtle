# Wygenerowanie klucza ssh

1.  w terminalu `ssh-keygen -t ed25519 -a 100 -C "adres_email"`
2. * podać własny email
3. ustawić własny passphrase (hasło do klucza) (opcjonalne)
4. wyświetlenie klucza ` cat ~/.ssh/id_ed25519.pub"`
5. * to `.pub` jest ważne
6. skopiować cały wiersz i wrzucić na plgrida

# Połączenie z Plgridem
1. Portal PLGrid → Moje usługi → aktywować Cyfronet Ares i WCSS LEM
2. ` ssh nazwa_uzytkownika@ares.cyfronet.pl`

# Przesyłanie plików:
1. Wskazać katalog lokalny
2. Wysłać na PLGrid: `scp -r "C:\Users\rafal\PycharmProjects\crypto-turtle\rl-module\services\rl"
    plgniemyslniepytaj@ares.cyfronet.pl:~`


# RSYNC
- rozwiązanie niezalecane
1. `wsl --install -d Ubuntu`
2. `sudo apt update`
3. `sudo apt install rsync`
4. `eval "$(ssh-agent -s)"`
5. `ssh-add /mnt/c/Users/rafal/.ssh/id_ed25519`
6. i potem normalnie `wsl`
7. ` rsync -avzP rl-module plgniemyslniepytaj@ares.cyfronet.pl:~/projekt`

# MobaXTerm
- rozwiązanie zalecane
- po zalogowaniu znajdujesz się w swoim katalogu `$HOME`
- poprostu przeciągasz tam swój moduł (cały) lub więcej
- ! nie przeciągasz całego projektu - tylko każdy moduł osobno
- wchodzisz w katalog `$SCRATCH` - i wrzucasz tam shared (może nie wszystko bo długo się przesyła ale to co potrzebne, np: układ folderów i tylko potrzebne pliki)
- Scratch: `"/net/afscra/people/plgniemyslniepytaj"` - twoja nazwa użytkownika
# Najpierw należy zainstalować środowisko conda
- ```conda create -n rl python=3.12 -y``` - rl (nazwa środowiska dowolna)
- ```conda activate rl```
- jeśli torch to taka instalacja: `pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
`
- inne requierments np: ` pip install pandas==2.3.3^C` 
- podczas procesu może być potrzbne jeszcze `eval "$(conda shell.bash hook)"`
- chatgpt pomoże jak coś
- instalowanie czegoś powoduje instalowanie innych potrzebnych bibliotek dla konkretnej biblioteki więc można sprawdzać na bieżąco `pip freeze`


# Skrypt `job.sh`
```
#!/bin/bash
#SBATCH --job-name=train_rl_1
#SBATCH --time=00:10:00
#SBATCH --account=plgcryptoturtles-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output="logs/joblog-%j.txt"
#SBATCH --error="errors/joberr-%j.txt"

module load miniconda3/24.5.0-0
eval "$(conda shell.bash hook)"
conda activate rl


# uruchomienie treningu
python ~/plgrid_train_agent_basic.py
```


# Odpalenie skryptu
1. Sprawdzenie kodu lokalnie w wsl `python3 rl-module/services/rl/plgrid_train_start.py`
2. Wysłanie do kolejki `sbatch job.sh`
3. Sprawdzenie kolejki `squeue -u plgniemyslniepytaj
4. Wyniki zapiszą się w folderze logs, errors (najpierw trzeba utworzyć) i w shared 
`
5. polecam przeczytać dokumentację: polecenia hpc-fs itp. 