# Pure-Rust-AI
This is simply my take on creating an AI from descriptions and tutorials I read about the theory in Rust

# How to use:
1. You can either run the precompiled exe or use the rust toolchain and Cargo to build and run the main file found in src
1. If you are compiling it yourself, then be aware there will be a lot of warnings, this is because this version has a lot of unused code as it is meant for testing the ready made network
1. You will have to select a number between 0 and 9999, these are the samples the network has been trained on, an ascii picture of the sample will be displayed with text saying what it's meant to represent
1. The program will also display the actual output of the AI as well as the interpreted result ("the network thinks it's a...:") as well as the cost value and time to execute

Please be aware the AI is not perfect. I haven't tested its accuracy but I'd say it's around 85% missing a few samples. In this case try again with another sample number! :)
