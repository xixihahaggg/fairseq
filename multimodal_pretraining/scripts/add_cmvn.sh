. ./path.sh
feature_dir=$1
for i in $feature_dir/raw_fbank*.scp; do # Whitespace-safe but not recursive.
    echo "$i"
    name=${i/scp/ark_after_cmvn}
    apply-cmvn --utt2spk=ark:/home/xixihahaggg/Dataset/how2/fbank_pitch_181506/utt2spk scp:/home/xixihahaggg/Dataset/how2/fbank_pitch_181506/cmvn_all_181506.scp_reformat scp:$i ark:$name
done


