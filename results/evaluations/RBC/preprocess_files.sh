echo "" > ./preprocessed_results.txt
for file in ./*/*.csv; do
   python3 preprocess.py $file >> ./preprocessed_results.txt
done