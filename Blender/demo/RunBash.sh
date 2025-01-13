#!/bin/bash

#Go to the directory RBE549
cd ../../../RBE549/nilampooranan_p3/Output/bbox/

#read the files name in loop
for file in *.txt
do
    echo $file
    echo ${file%_bb.txt}
    #keep count of line number
    linecount=0
    # rm -rf ../../../../Git_repos/I2L-MeshNet_RELEASE/demo/input.jpg
    cp ../inputs/frame${file%_bb.txt}.jpg ../../../../Git_repos/I2L-MeshNet_RELEASE/demo/input.jpg
    while IFS=' ' read -r xmin ymin width height
    do
        #keep count of the line number that is read

        linecount=$((linecount+1))
        if [ $linecount -eq 1 ]
        then
            cd ../../../../Git_repos/I2L-MeshNet_RELEASE/demo/
        fi
        # python3 demo.py --gpu 0 --stage param --test_epoch 12 --xmin $xmin --ymin $ymin --width $width --height $height
        #done
        #move filename of same name as ${file%_bb.txt} to the directory
        echo $xmin $ymin $width $height $linecount
        

        python3 demo.py --gpu 0 --stage param --test_epoch 12 --xmin $xmin --ymin $ymin --width $width --height $height
        mv ./output_mesh_lixel.obj ../../../RBE549/nilampooranan_p3/Data/Assets/obj/human_models/${file%_bb.txt}_$linecount.obj
        
    done < "$file"
    cd ../../../RBE549/nilampooranan_p3/Output/bbox/
#end
done

