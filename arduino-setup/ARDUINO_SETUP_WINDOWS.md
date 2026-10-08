# Connecting an Arduino UNO Q to GitHub (Windows)

The Windows counterpart to [`ARDUINO_SETUP_MAC.md`](../ARDUINO_SETUP_MAC.md).
Same eight steps, with the commands that actually differ on Windows.

Commands marked **Windows** run on your PC.
Commands marked **UNO Q** run on the board, after you `ssh arduino@<yourboardIP>` into it.
The UNO Q runs Linux, so every **UNO Q** command is identical to the Mac guide's.

Find the board IP in Arduino App Lab → Settings → Network Connections.

## Which terminal?

**Use Git Bash** (installed with [Git for Windows](https://git-scm.com/download/win)).
It ships `ssh-copy-id` and `unzip`, which Windows itself does not, so the Mac
instructions mostly work unchanged. Open it from the Start menu, or right-click a
folder → **Open Git Bash here**.

PowerShell works too, and where the two differ this guide gives both. Just don't
mix them up mid-step — PowerShell has no `&&`, no `source`, and no `ssh-copy-id`.

Both share the same key, because Git Bash's `~` *is* `C:\Users\<you>`, which is
where Windows' own SSH looks. One key, both tools.

> No git yet? Run `winget install --id Git.Git -e` in PowerShell, or download it
> from the link above. Reopen the terminal afterwards so the PATH updates.

---

## 1. SSH from Windows into the UNO Q

**Windows (Git Bash)**

```bash
# Make a key (press Enter for the defaults; a passphrase is optional)
ssh-keygen -t ed25519

# Copy it to the board. This is why we use Git Bash -- ssh-copy-id comes with
# it, but is NOT part of Windows' built-in SSH.
ssh-copy-id arduino@<yourboardIP>

# Check that it logs in without asking for a password
ssh arduino@<yourboardIP>
```

**Windows (PowerShell)** — if you'd rather not use Git Bash, replace the
`ssh-copy-id` line with this, which does the same job by hand:

```powershell
ssh-keygen -t ed25519
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | ssh arduino@<yourboardIP> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
ssh arduino@<yourboardIP>
```

## 2. Set up git on the UNO Q

Identical to the Mac guide — this runs on the board, which is Linux.

**UNO Q**

```bash
git --version || sudo apt install -y git
git config --global user.name "MyName UNO Q"
git config --global user.email "youremail@yourdomain.com"
```

## 3. Give the UNO Q its own key for GitHub

The key from step 1 lets *your PC* log into *the board*. It does not let the
board talk to GitHub — the board needs its own key for that.

**UNO Q**

```bash
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519   # make one if missing
cat ~/.ssh/id_ed25519.pub                           # copy the whole line it prints
```

Select the line in the terminal window and copy it with **Ctrl + Shift + C** — in
Git Bash, plain Ctrl+C interrupts the running command instead of copying.

On github.com: your icon (top right) → **Settings** → **SSH and GPG keys** →
**New SSH key**. Paste it, title it "UNO Q", save. GitHub may ask for your
password or 2FA code — adding a key is a security event, and that key then acts
as you.

**UNO Q**

```bash
ssh -T git@github.com    # type "yes" the first time; should say "Hi <username>!"
```

## 4. Push the board's ArduinoApps to GitHub

Create an **empty** repo named `ArduinoApps` first (no README, no .gitignore).
Plain `git` cannot create a repository on GitHub — only the website or the `gh`
tool can:

**Windows**

```powershell
gh repo create ArduinoApps --public     # or create it on github.com by hand
```

Then, on the board:

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

Copy-pasted someone else's username into that remote line? Fix it, then re-run
the push:

```bash
git remote set-url origin git@github.com:<yourusername>/ArduinoApps.git
```

## 5. Clone ArduinoApps to your PC

Cloning over SSH needs your **PC's** key on GitHub too — step 1 created it, but
step 3 only added the board's.

**Windows (PowerShell)**

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard
```

**Windows (Git Bash)**

```bash
cat ~/.ssh/id_ed25519.pub | clip
```

Add it on GitHub as a second key, titled "Windows PC". Then clone:

```powershell
cd "$env:USERPROFILE\Documents"        # or wherever you keep projects
git clone git@github.com:<yourusername>/ArduinoApps.git
```

Two easier options if SSH is being stubborn: clone over HTTPS with
`gh repo clone <yourusername>/ArduinoApps`, which reuses your existing `gh`
login and needs no key at all, or use GitHub Desktop.

> **Quote any path containing spaces.** `cd C:\Users\you\My Folder` fails;
> `cd "C:\Users\you\My Folder"` works.

## 6. Add the course files

Download the zip, or pull it into your fork of Chris' repository.

**Windows (PowerShell)**

```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\<thezipfile>.zip" -DestinationPath "$env:USERPROFILE\Documents\ArduinoApps"
```

**Windows (Git Bash)**

```bash
unzip ~/Downloads/<thezipfile>.zip -d ~/Documents/ArduinoApps/
```

Right-click → **Extract All…** in File Explorer works just as well. Afterwards
check you didn't end up with `ArduinoApps\thezipfile\...` — the files should sit
directly in `ArduinoApps`, not nested one folder deeper.

## 7. Run deploy.py and set the board IP

Open the `ArduinoApps` folder in VS Code and run `deploy.py`. **On Windows the
command is `python`, not `python3`** — `python3` either doesn't exist or opens
the Microsoft Store:

**Windows**

```powershell
cd "$env:USERPROFILE\Documents\ArduinoApps"
python deploy.py
```

That creates `board_secrets.py`. Open it and put your board's IP in **both**
places in the file.

> If VS Code reports `ModuleNotFoundError` for something you know is installed,
> it's running a different Python than your terminal: Ctrl+Shift+P →
> **Python: Select Interpreter**, and pick the one you installed into. Watch for
> a `(.venv)` prefix on your prompt too — while a virtual environment is active,
> `python` means *that* environment's Python, not the system one.

## 8. Keep board_secrets.py out of git

It holds your board's address and credentials, so it must never be committed.

**Windows (Git Bash)**

```bash
grep -qx 'board_secrets.py' .gitignore || echo 'board_secrets.py' >> .gitignore
git check-ignore -v board_secrets.py          # should print the .gitignore rule
git rm --cached board_secrets.py 2>/dev/null  # only if it was committed earlier
git add .gitignore
git commit -m "Ignore board_secrets.py"
git push
```

**Windows (PowerShell)**

```powershell
if (-not (Select-String -Path .gitignore -Pattern '^board_secrets\.py$' -Quiet)) {
    Add-Content .gitignore 'board_secrets.py' -Encoding utf8
}
git check-ignore -v board_secrets.py
git rm --cached board_secrets.py
git add .gitignore
git commit -m "Ignore board_secrets.py"
git push
```

`git check-ignore` printing a rule is the confirmation that it really is
ignored. Silence means it is **not** — fix that before you push.

Can't see `.gitignore` in File Explorer? **View → Show → Hidden items.**

---

## Windows vs. Linux / Mac

| Task | UNO Q (Linux) | Mac | Windows |
|---|---|---|---|
| Install git | `sudo apt install -y git` | `xcode-select --install` | `winget install --id Git.Git -e` |
| Copy a key to another machine | — | `ssh-copy-id` | `ssh-copy-id` (Git Bash only) |
| Copy a file to the clipboard | — | `pbcopy < file` | `Get-Content file \| Set-Clipboard`, or `cat file \| clip` |
| Run Python | `python3` | `python3` | `python` |
| Activate a venv | `source .venv/bin/activate` | same | `.venv\Scripts\Activate.ps1` |
| Unzip | `unzip` | `unzip` | `Expand-Archive`, or `unzip` in Git Bash |
| Chain two commands | `a && b` | same | `a; if ($?) { b }` in PowerShell |
| Show hidden files | `ls -a` | Cmd + Shift + . | View → Show → Hidden items |

## Windows-only snags

**Line endings.** Windows ends lines with CRLF, Linux with LF. Code you edit on
your PC and push to the board can arrive with stray `\r`, which surfaces as
`bad interpreter: No such file or directory` on a script that looks perfectly
fine. Because this repo's files run on a Linux board, set this once:

```powershell
git config --global core.autocrlf input
```

That keeps LF in commits while letting you edit normally. The
`warning: LF will be replaced by CRLF` messages git prints are harmless.

**"Permissions for 'id_ed25519' are too open."** Windows' SSH refuses a private
key other accounts can read. Lock it to just you:

```powershell
icacls "$env:USERPROFILE\.ssh\id_ed25519" /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

**`ssh: command not found`.** You're in an old Command Prompt, or Git isn't on
the PATH. Use Git Bash or PowerShell — Windows 10 and 11 include OpenSSH as
standard, and `ssh -V` confirms it.

**The board drops off the network.** Its IP can change when it reconnects. If
`ssh` suddenly times out, re-check the address in Arduino App Lab → Settings →
Network Connections, and update `board_secrets.py` if it moved.
