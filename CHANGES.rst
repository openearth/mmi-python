Changes
-------

- Added support for single values for send/recv_array.

- Arrays of length 0 are now handled correctly.

- Support for arrays being chopped in chunks. Both sides must have the same
  mmi version.

- recv_array now supports polling for timeout.

- Added MMIClient.

- Added tracker client.
