REM COMMAND LINE MS BUILD
REM Note: /m:2 = 2 threads, but for host code only...
REM
REM /FS is required alongside /m: without it several cl.exe write the same
REM vc143.pdb and fail with C1041. The link then falls back to objects from an
REM earlier build, so it can still produce a ccminer.exe while msbuild reports
REM failure -- a binary with stale code and a non-zero exit nobody read. The CL
REM environment variable appends the option to every cl invocation without
REM overriding the per-file AdditionalOptions in the project.
set CL=/FS

msbuild ccminer.vcxproj /m /p:Configuration=Release
