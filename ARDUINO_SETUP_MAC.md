# Connecting an Arduino UNO Q to GitHub (macOS)

Commands marked **Mac** run in the Mac Terminal app (Applications → Utilities → Terminal).
Commands marked **UNO Q** run on the board, after you `ssh arduino@<yourboardIP>` into it.

Find the board IP in Arduino App Lab → Settings → Network Connections.

---

## 1. SSH from your Mac into the UNO Q

**Mac**
```bash
# Make a key (press Enter to accept the defaults; a passphrase is optional)
ssh-keygen -t ed25519

# Copy it to the board. macOS ships ssh-copy-id, so this works as-is.
ssh-copy-id arduino@<yourboardIP>

# Check that it logs in without asking for a password
ssh arduino@<yourboardIP>
```

## 2. Set up git on the UNO Q

**UNO Q**
```bash
git --version || sudo apt install -y git
git config --global user.name "MyName UNO Q"
git config --global user.email "youremail@yourdomain.com"
```

## 3. Give the UNO Q its own key for GitHub

The key you made on the Mac lets the Mac log into the board. The board needs its
own key to push to GitHub.

**UNO Q**
```bash
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519   # make one if missing
cat ~/.ssh/id_ed25519.pub                           # copy the whole line it prints
```

On github.com: your icon (top right) → **Settings** → **SSH and GPG keys** →
**New SSH key**. Paste the key, give it a title like "UNO Q", and save.
GitHub may ask for your password or 2FA code.

Check that it works:

**UNO Q**
```bash
ssh -T git@github.com    # type "yes" the first time; should say "Hi <username>!"
```

## 4. Push the board's ArduinoApps to GitHub

First, on github.com, create an **empty** repository named `ArduinoApps`
(no README, no .gitignore). git can't create the repository on GitHub for you.

**UNO Q**
```bash
cd ~/ArduinoApps
printf '.cache/\n__pycache__/\n*.pyc\n.DS_Store\nboard_secrets.py\n' > .gitignore
git init -b main
git add .
git status        # make sure no .cache folders are listed
git commit -m "Initial commit from my UNO Q"
git remote add origin git@github.com:<yourusername>/ArduinoApps.git
git push -u origin main
```

If you added the wrong remote, fix it and then run the push again:
```bash
git remote set-url origin git@github.com:<yourusername>/ArduinoApps.git
```

## 5. Clone ArduinoApps to your Mac

To clone over SSH, GitHub also needs your **Mac's** key:

**Mac**
```bash
pbcopy < ~/.ssh/id_ed25519.pub     # copies the Mac key to the clipboard
```
Add it on GitHub as another SSH key (title it "Mac"), then:

**Mac**
```bash
xcode-select --install 2>/dev/null   # installs git if the Mac doesn't have it yet
cd ~/Documents                       # or wherever you keep projects
git clone git@github.com:<yourusername>/ArduinoApps.git
```

You can also skip the Mac key and clone with **GitHub Desktop** instead.

## 6. Add the course files

Download the zip, or pull the changes into your fork of Chris' repository. Then:

**Mac**
```bash
unzip ~/Downloads/<thezipfile>.zip -d ~/Documents/ArduinoApps/
```
(Double-clicking the zip in Finder works too. Then drag the folder into `ArduinoApps`.)

## 7. Run deploy.py and set the board IP

Open the `ArduinoApps` folder in VS Code and run `deploy.py`. On a Mac use
`python3`, not `python`:

**Mac**
```bash
cd ~/Documents/ArduinoApps
python3 deploy.py
```

That creates `board_secrets.py`. Open it and put your board's IP address in
**both** places in the file.

## 8. Keep board_secrets.py out of git

**Mac**
```bash
grep -qx 'board_secrets.py' .gitignore || echo 'board_secrets.py' >> .gitignore
git check-ignore -v board_secrets.py   # should print the .gitignore rule
git rm --cached board_secrets.py 2>/dev/null   # only needed if it was committed earlier
git add .gitignore
git commit -m "Ignore board_secrets.py"
git push
```

### Mac vs. Linux differences
| Task | Linux / UNO Q | Mac |
|---|---|---|
| Install git | `sudo apt install -y git` | `xcode-select --install` (or `brew install git`) |
| Copy a file to the clipboard | select it and copy | `pbcopy < file` |
| Run Python | `python` / `python3` | `python3` |
| Unzip | `unzip` | `unzip`, or double-click in Finder |
| Hidden files in Finder | — | Cmd + Shift + . to show `.gitignore` |
