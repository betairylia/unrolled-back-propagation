:Loop
python unrolled_toy.py unroll_heavy_test_adam\ -gpu 2 -u 2 -l 18 -sig 0.002 -avg 3 -data 65536 -nn 16 16 16 16 16 16 16 16 8 8 -adam -lrelu
if %errorlevel% neq 0 goto :Loop
echo Complete.