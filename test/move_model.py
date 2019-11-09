#!/usr/bin/python
import argparse
import os

def count_tiles(net):
    t_count = 0
    while os.path.isfile(net + "-tile" + str(t_count) + ".puma"):
        t_count -= -1
    return t_count

def check_model(net):
    if not os.path.isfile(net+"-tile0.puma"):
        exit(""+net+"-tile0.puma not found.")
    
    t_count = count_tiles(net)
    print("Model " + net + " has " + str(t_count) + " tiles")

def move_tiles_instructions(tile_instr_file, offset):
    new_instructions = ""
    with open(tile_instr_file, 'r') as f:
        for line in f:
            if "target_addr=" in line:
                # print(line)
                target_addr_idx = line.find("target_addr=")+12
                comma_idx = line.find(",",target_addr_idx)
                old_target = line[target_addr_idx:comma_idx]
                line = line.replace("target_addr="+old_target,"target_addr="+str(int(old_target)+offset))
                # print(line)
            if "vtile_id=" in line:
                # print(line)
                vtile_id_idx = line.find("vtile_id=")+9
                comma_idx = line.find(",",vtile_id_idx)
                old_vtile_id = line[vtile_id_idx:comma_idx]
                line = line.replace("vtile_id="+old_vtile_id,"vtile_id="+str(int(old_vtile_id)+offset))
                # print(line)
            new_instructions += line
    
    with open(tile_instr_file, 'w') as f:
        f.write(new_instructions)

def move_tiles(net, offset):
    t_count = count_tiles(net)
    offset = int(offset)
    assert (offset > 0)
    for t in range(t_count-1,-1,-1): # move backwards to avoid overwriting files
        old_name = net + "-tile" + str(t) + ".puma"
        new_name = net + "-tile" + str(t+offset) + ".puma"
        os.rename(old_name, new_name)
        move_tiles_instructions(new_name,offset)
        for c in range(8):
            old_name = net + "-tile" + str(t) + "-core" + str(c) + ".puma"
            new_name = net + "-tile" + str(t+offset) + "-core" + str(c) + ".puma"
            os.rename(old_name, new_name)
            for m in range(2):
                old_name = net + "-tile" + str(t) + "-core" + str(c) + "-mvmu" + str(m) + ".weights"
                if os.path.isfile(old_name):
                    new_name = net + "-tile" + str(t+offset) + "-core" + str(c) + "-mvmu" + str(m) + ".weights"
                    os.rename(old_name, new_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "net", help="The net name")
    parser.add_argument(
        "offset", help="The offset to be add to tiles number")
    args = parser.parse_args()
    
    check_model(args.net)
    move_tiles(args.net, args.offset)
    print("Done :)")