#!/usr/bin/env bash
# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# hf_mirror endpoint
export HF_ENDPOINT="https://hf-mirror.com"

trap 'printf "${YELLOW}\nDownload interrupted. If you re-run the command, you can resume the download from the breakpoint.\n${NC}"; exit 1' INT

display_help() {
    cat << EOF
Usage:
  hfd <model_id> [--include include_pattern] [--exclude exclude_pattern] [--hf_username username] [--hf_token token] [--tool wget|aria2c] [-x threads] [--dataset]

Description:
  Downloads a model or dataset from HuggingFace using the provided model ID.

Parameters:
  model_id        The HuggingFace model ID in the format 'repo/model_name'.
  --include       (Optional) Flag to specify a string pattern to include files for downloading.
  --exclude       (Optional) Flag to specify a string pattern to exclude files from downloading.
  exclude_pattern The pattern to match against filenames for exclusion.
  --hf_username   (Optional) HuggingFace username for authentication.
  --hf_token      (Optional) HuggingFace token for authentication.
  --tool          (Optional) Download tool to use. Can be wget or aria2c (default).
  -x              (Optional) Number of download threads for aria2c.
  --save_dir     (Optional) Flag to specify a string path to download files.
  --dataset       (Optional) Flag to indicate downloading a dataset.

Example:
  hfd bigscience/bloom-560m --exclude safetensors
  hfd meta-llama/Llama-2-7b --hf_username myuser --hf_token mytoken --tool aria2c -x 8 --save_dir your_dir
  hfd lavita/medical-qa-shared-task-v1-toy --save_dir your_dir --dataset
EOF
    exit 1
}

MODEL_ID=$1
shift

# Default values
TOOL="aria2c"
THREADS=1
HF_ENDPOINT=${HF_ENDPOINT:-"https://huggingface.co"}

while [[ $# -gt 0 ]]; do
    case $1 in
        --include) INCLUDE_PATTERN="$2"; shift 2 ;;
        --exclude) EXCLUDE_PATTERN="$2"; shift 2 ;;
        --hf_username) HF_USERNAME="$2"; shift 2 ;;
        --hf_token) HF_TOKEN="$2"; shift 2 ;;
        --tool) TOOL="$2"; shift 2 ;;
        -x) THREADS="$2"; shift 2 ;;
        --save_dir) SAVE_DIR="${2:-./}"; shift 2 ;;
        --dataset) DATASET=1; shift ;;
        *) shift ;;
    esac
done

# Check if aria2, wget, curl, git, and git-lfs are installed
check_command() {
    if ! command -v $1 &>/dev/null; then
        echo -e "${RED}$1 is not installed. Please install it first.${NC}"
        exit 1
    fi
}

[[ "$TOOL" == "aria2c" ]] && check_command aria2c
[[ "$TOOL" == "wget" ]] && check_command wget
check_command curl; check_command git; check_command git-lfs

[[ -z "$MODEL_ID" || "$MODEL_ID" =~ ^-h ]] && display_help

if [[ -n "$SAVE_DIR" ]]; then
    mkdir -p $SAVE_DIR
    echo "Downloading to ./$SAVE_DIR"
else
    SAVE_DIR="./"
fi

SAVE_DIR="$SAVE_DIR/${MODEL_ID#*/}"

if [[ "$DATASET" == 1 ]]; then
    MODEL_ID="datasets/$MODEL_ID"
fi

if [ -d "$SAVE_DIR/.git" ]; then
    printf "${YELLOW}%s exists, Skip Clone.\n${NC}" "$SAVE_DIR"
    cd "$SAVE_DIR" && GIT_LFS_SKIP_SMUDGE=1 git pull || { printf "Git pull failed.\n"; exit 1; }
else
    REPO_URL="$HF_ENDPOINT/$MODEL_ID"
    GIT_REFS_URL="${REPO_URL}/info/refs?service=git-upload-pack"
    echo "Test GIT_REFS_URL: $GIT_REFS_URL"
    response=$(curl -s -o /dev/null -w "%{http_code}" "$GIT_REFS_URL")
    if [ "$response" == "401" ] || [ "$response" == "403" ]; then
        if [[ -z "$HF_USERNAME" || -z "$HF_TOKEN" ]]; then
            printf "${RED}HTTP Status Code: $response.\nThe repository requires authentication, but --hf_username and --hf_token is not passed. Please get token from https://huggingface.co/settings/tokens.\nExiting.\n${NC}"
            exit 1
        fi
        REPO_URL="https://$HF_USERNAME:$HF_TOKEN@${HF_ENDPOINT#https://}/$MODEL_ID"
    elif [ "$response" != "200" ]; then
        echo -e "${RED}Unexpected HTTP Status Code: $response.\nExiting.\n${NC}"; exit 1
    fi
    echo "git clone $REPO_URL"

    GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO_URL" "$SAVE_DIR" && cd "$SAVE_DIR" || { printf "${RED}Git clone failed.\n${NC}"; exit 1; }
    for file in $(git lfs ls-files | awk '{print $3}'); do
        truncate -s 0 "$file"
    done
fi

printf "\nStart Downloading lfs files, bash script:\n"
files=$(git lfs ls-files | awk '{print $3}')
declare -a urls

for file in $files; do
    url="$HF_ENDPOINT/$MODEL_ID/resolve/main/$file"
    file_dir=$(dirname "$file")
    mkdir -p "$file_dir"
    if [[ "$TOOL" == "wget" ]]; then
        download_cmd="wget --no-check-certificate -c \"$url\" -O \"$file\""
        [[ -n "$HF_TOKEN" ]] && download_cmd="wget --no-check-certificate --header=\"Authorization: Bearer ${HF_TOKEN}\" -c \"$url\" -O \"$file\""
    else
        download_cmd="aria2c --check-certificate=false -x $THREADS -s $THREADS -k 1M -c \"$url\" -d \"$file_dir\" -o \"$(basename "$file")\""
        [[ -n "$HF_TOKEN" ]] && download_cmd="aria2c --check-certificate=false --header=\"Authorization: Bearer ${HF_TOKEN}\" -x $THREADS -s $THREADS -k 1M -c \"$url\" -d \"$file_dir\" -o \"$(basename "$file")\""
    fi
    [[ -n "$INCLUDE_PATTERN" && $file != *"$INCLUDE_PATTERN"* ]] && printf "# %s\n" "$download_cmd" && continue
    [[ -n "$EXCLUDE_PATTERN" && $file == *"$EXCLUDE_PATTERN"* ]] && printf "# %s\n" "$download_cmd" && continue
    printf "%s\n" "$download_cmd"
    urls+=("$url|$file")
done

for url_file in "${urls[@]}"; do
    IFS='|' read -r url file <<< "$url_file"
    file_dir=$(dirname "$file")
    if [[ "$TOOL" == "wget" ]]; then
        [[ -n "$HF_TOKEN" ]] && wget --no-check-certificat --header="Authorization: Bearer ${HF_TOKEN}" -c "$url" -O "$file" || wget --no-check-certificat -c "$url" -O "$file"
    else
        [[ -n "$HF_TOKEN" ]] && aria2c --check-certificate=false --header="Authorization: Bearer ${HF_TOKEN}" -x $THREADS -s $THREADS -k 1M -c "$url" -d "$file_dir" -o "$(basename "$file")" || aria2c --check-certificate=false -x $THREADS -s $THREADS -k 1M -c "$url" -d "$file_dir" -o "$(basename "$file")"
    fi
    [[ $? -eq 0 ]] && printf "Downloaded %s successfully.\n" "$url" || { printf "${RED}Failed to download %s.\n${NC}" "$url"; exit 1; }
done

printf "${GREEN}Download completed successfully.\n${NC}"
