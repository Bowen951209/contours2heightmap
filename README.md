This is the directory for storing benchmark results.
Because the performances I am comparing ranges wide,
from several minutes to less than a second for a big input image,
this is considered **macro-benchmarking**. Thus micro-benchmarking
tools such as `criternion` doesn't fit.

Becuase the measurement is rather simple, I choose to do it manually.
This is the procedure:
 - Target fuuction: `linear_fill`.
 - Checkout a target revision.
 - Run in *release mode*.
 - Warm up (run the program without sampling) once before taking samples.
 - Run the program with input images:
    * `one_hill.jpg`
    * `two_hills.jpg`
    * `turtle.jpg`
 - For each image, run the program 5 times and record
    * Execution Time: Measured internally using Rust's `std::time::Instant`.
    * Peak Memory Usage: Measured externally using the Linux `time` command.
 - Fill the data/information in the universal CSV file.
