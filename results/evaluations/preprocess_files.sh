./Disc_Red_Action_Space/preprocess_files.sh
cat ./Disc_Red_Action_Space/preprocessed_results.txt > ./preprocessed_results.txt

for file in ./*/*.csv; do
   python3 preprocess.py $file >> ./preprocessed_results.txt
done

./RBC/preprocess_files.sh
cat ./RBC/preprocessed_results.txt >> ./preprocessed_results.txt

./Extra/preprocess_files.sh
cat ./Extra/preprocessed_results.txt >> ./preprocessed_results.txt
