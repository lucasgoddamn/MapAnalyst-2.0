param(
  [string]$SourceBranch = "main",
  [string]$ExportBranch = "github-main",
  [string]$GitHubRemoteName = "github",
  [string]$GitHubRemoteUrl = "https://github.com/lucasgoddamn/MapAnalyst-2.0.git",
  [string]$GitHubTargetBranch = "main"
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
  param([string]$WorkDir)
  $GitArgs = @()
  foreach ($a in $args) {
    if ($null -eq $a) { continue }
    if ($a -is [System.Array]) {
      foreach ($nested in $a) {
        if ($null -ne $nested) {
          $GitArgs += [string]$nested
        }
      }
    } else {
      $GitArgs += [string]$a
    }
  }
  if (-not $GitArgs -or $GitArgs.Count -eq 0) {
    throw "Invoke-Git called without git arguments."
  }
  $cmdText = if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    "git " + ($GitArgs -join " ")
  } else {
    "git -C `"$WorkDir`" " + ($GitArgs -join " ")
  }
  Write-Host ">> $cmdText" -ForegroundColor Cyan
  if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    & git @GitArgs
  } else {
    & git -C $WorkDir @GitArgs
  }
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed: $cmdText"
  }
}

# Ensure we are inside a git repository
$insideRepo = (& git rev-parse --is-inside-work-tree 2>$null).Trim()
if ($insideRepo -ne "true") {
  throw "Not inside a git repository."
}
$repoRoot = (& git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
  throw "Could not determine repository root."
}

# Require clean working tree to ensure deterministic export
$porcelain = & git -C $repoRoot status --porcelain
if ($porcelain) {
  throw "Working tree is not clean. Commit/stash changes first."
}

# Validate source branch exists locally
& git -C $repoRoot show-ref --verify --quiet "refs/heads/$SourceBranch"
if ($LASTEXITCODE -ne 0) {
  throw "Source branch '$SourceBranch' does not exist locally."
}

# Ensure GitHub remote exists (or add/update it) without triggering stderr failure
$remoteNames = @(& git -C $repoRoot remote) | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$remoteExists = $remoteNames -contains $GitHubRemoteName
if (-not $remoteExists) {
  Invoke-Git $repoRoot @("remote", "add", $GitHubRemoteName, $GitHubRemoteUrl)
} else {
  $remoteUrl = (& git -C $repoRoot remote get-url $GitHubRemoteName).Trim()
  if ($remoteUrl -ne $GitHubRemoteUrl) {
    Write-Warning "Remote '$GitHubRemoteName' points to '$remoteUrl'. Updating to '$GitHubRemoteUrl'."
    Invoke-Git $repoRoot @("remote", "set-url", $GitHubRemoteName, $GitHubRemoteUrl)
  }
}

Invoke-Git $repoRoot @("fetch", $GitHubRemoteName)

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$worktreeDir = Join-Path ([System.IO.Path]::GetTempPath()) "mapanalyst-github-export-$stamp-$PID"

try {
  # Create isolated checkout from source branch (prevents file-lock issues in active working tree)
  Invoke-Git $repoRoot @("worktree", "add", "--force", $worktreeDir, $SourceBranch)

  # Check if export branch already exists locally
  & git -C $repoRoot show-ref --verify --quiet "refs/heads/$ExportBranch"
  $exportExists = ($LASTEXITCODE -eq 0)

  if ($exportExists) {
    Invoke-Git $worktreeDir @("switch", "--ignore-other-worktrees", $ExportBranch)
  } else {
    Invoke-Git $worktreeDir @("switch", "--orphan", $ExportBranch)
  }

  # Recreate export branch contents from source branch, then remove private folders
  Invoke-Git $worktreeDir @("rm", "-r", "-f", "--ignore-unmatch", ".")
  Invoke-Git $worktreeDir @("checkout", $SourceBranch, "--", ".")
  Invoke-Git $worktreeDir @("rm", "-r", "-f", "--ignore-unmatch", "TextMa", "VortragMa")
  Invoke-Git $worktreeDir @("add", "-A")

  & git -C $worktreeDir diff --cached --quiet
  if ($LASTEXITCODE -eq 0) {
    Write-Host "No changes to export." -ForegroundColor Yellow
  } else {
    $msg = "Sync GitHub export from $SourceBranch (exclude TextMa, VortragMa)"
    Invoke-Git $worktreeDir @("commit", "-m", $msg)
  }

  # Push export branch to GitHub target branch
  $remoteBranchOut = & git -C $worktreeDir ls-remote --heads $GitHubRemoteName $GitHubTargetBranch
  if ($null -eq $remoteBranchOut) {
    $remoteBranch = ""
  } else {
    $remoteBranch = (($remoteBranchOut -join "`n")).Trim()
  }
  if ([string]::IsNullOrWhiteSpace($remoteBranch)) {
    Invoke-Git $worktreeDir @("push", "-u", $GitHubRemoteName, "$ExportBranch`:$GitHubTargetBranch")
  } else {
    Invoke-Git $worktreeDir @("push", "--force-with-lease", $GitHubRemoteName, "$ExportBranch`:$GitHubTargetBranch")
  }
}
finally {
  # Best-effort cleanup for the temporary worktree
  if (Test-Path -LiteralPath $worktreeDir) {
    try {
      Invoke-Git $repoRoot @("worktree", "remove", "--force", $worktreeDir)
    } catch {
      Write-Warning "Could not fully remove temporary worktree via git: $($_.Exception.Message)"
    }
    if (Test-Path -LiteralPath $worktreeDir) {
      Remove-Item -LiteralPath $worktreeDir -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
}

Write-Host "Done. GitHub sync complete." -ForegroundColor Green
