#!/usr/bin/env bash

# select dataset (defaults to train)
if [ $#  = 1 ] && [[ "train dev test" =~ $1 ]]; then
    ver=$1;
else
    ver=train
fi

# download training data
curl -o data.zip -L https://github.com/lovit/namuwikitext/releases/download/v0.3/namuwikitext_20200302.$ver.zip

# unzip compressed file
unzip data.zip
rm data.zip

# filter data
echo filtering
grep -v -e '\\' -e '\[' -e =.*=$ -e http -e 'www\.' -e ^[[:space:]]*$ namuwikitext_20200302.$ver |
sed -e 's/0/ 공 /g' -e 's/1/ 일 /g' -e 's/2/ 이 /g' -e 's/3/ 삼 /g' -e 's/4/ 사 /g' -e 's/5/ 오 /g' -e 's/6/ 륙 /g' -e 's/7/ 칠 /g' -e 's/8/ 팔 /g' -e 's/9/ 구 /g' |
perl -CSD -pe 's/[^\p{L}\n]/ /g; s/[^\S\n]+/ /g; s/^\s+|[^\S\n]+$//mg;' |
grep -v -P '[^\p{Hangul}\s]' > filtered.txt
rm namuwikitext_20200302.$ver

# deconstruct 한글
echo deconstructing
python3 decompose_letters.py filtered.txt preprocessed.txt
rm filtered.txt