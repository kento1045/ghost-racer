#!/bin/sh
set -eu

project=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
betaflight_ref=a3495a384f801ffaedc188188c1cae85dad11f39
upstream="ghost-racer-upstream:a3495a384"
fingerprint=$(cat "$project/docker/Dockerfile" "$project"/docker/*.patch "$project"/docker/rotors/* "$0" | shasum -a 256 | cut -d ' ' -f 1)
if [ "${1-}" = "--ensure" ]; then
    installed=$(docker image inspect --format '{{index .Config.Labels "ghost-racer.build"}}' ghost-racer-sitl:local 2>/dev/null || true)
    [ "$fingerprint" != "$installed" ] || exit 0
fi
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

# The official recipe selects the amd64 Gazebo package repository explicitly.
recipe="$tmpdir/containerfile.gazebo"
curl --fail --silent --show-error --location --retry 3 \
    "https://raw.githubusercontent.com/betaflight/betaflight/$betaflight_ref/.devcontainer/containerfile.gazebo" \
    --output "$recipe"
echo "9085838b246b0cd3f767977b17e6c5ae4ce3f571c3e690fe6a3b6312e0ec7a25  $recipe" | shasum -a 256 -c -
docker build --platform linux/amd64 -t "$upstream" -f "$recipe" "$tmpdir"
docker build --platform linux/amd64 --label "ghost-racer.build=$fingerprint" --build-arg "UPSTREAM=$upstream" \
    --build-arg "BETAFLIGHT_REF=$betaflight_ref" -t ghost-racer-sitl:local "$project/docker"
