if [[ ! -o interactive ]]; then
    return
fi

compctl -K _oat oat

_oat() {
  local word words completions
  read -cA words
  word="${words[2]}"

  if [ "${#words}" -eq 2 ]; then
    completions="$(oat commands)"
  else
    completions="$(oat completions "${word}")"
  fi

  reply=("${(ps:\n:)completions}")
}
