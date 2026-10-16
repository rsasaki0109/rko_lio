#!/usr/bin/env bash
# Builds the given name:commit:kind triples one after another; logs per build.
for spec in "$@"; do
  IFS=: read -r name commit kind <<< "${spec}"
  log="/home/user/logs/build_${name}.log"
  /home/user/scripts/build_variant.sh "${name}" "${commit}" "${kind}" > "${log}" 2>&1
  rc=$?
  echo "exit=${rc}" >> "${log}"
  echo "$(date -u +%H:%M:%S) ${name} ${commit} ${kind} exit=${rc} $(grep -o 'BUILD_OK.*' "${log}" | tail -1)"
done
