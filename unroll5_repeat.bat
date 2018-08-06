:Loop
python unrolled_toy.py unroll5_test_adam\ -gpu 2 -lr 0.01 -alpha 0.9 -u 5 -l 5 -sig 0.03 -avg 5 -data 2048 -nn 20 20 20 20 -adam
if %errorlevel% neq 0 goto :Loop
echo Complete.