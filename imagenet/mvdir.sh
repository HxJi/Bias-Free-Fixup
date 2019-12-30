#!/bin/bash
for i in {1..100..2}
do
  mv activation-$i-* epoch-$i-batch-20/
done 
