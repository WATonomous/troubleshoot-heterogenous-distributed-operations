set -ex

# download boost
wget --quiet https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.bz2
# verify sha256
printf "6478edfe2f3305127cffe8caf73ea0176c53769f4bf1585be237eb30798c3b8e  boost_1_83_0.tar.bz2" | sha256sum -c

# extract
tar -xjf boost_1_83_0.tar.bz2

pushd boost_1_83_0

# Ref:
# - https://stackoverflow.com/a/70682496
# - https://www.boost.org/doc/libs/1_83_0/more/getting_started/unix-variants.html

./bootstrap.sh
./b2 cxxflags=-fPIC cflags=-fPIC install
popd
