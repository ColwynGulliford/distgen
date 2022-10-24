import json
import os
import copy

def main():

    exs = {}

    files = os.listdir('.')
    for filename in files:

        if(filename.endswith('json') and 'examples' not in filename):
            print(filename)
            with open(filename) as jid:
                data = json.load(jid)

            ndata = copy.deepcopy(data)

            ndata['total_charge']=data['beam']['params']['total_charge']
            ndata['n_particles']=data['generator']['rand']['count'] 
            ndata['random_type']=data['generator']['rand']['type'] 
            ndata['count']=data['generator']['rand']['count']

            del[ndata['generator']]
            del[ndata['beam']]

            exs[filename[:-8]]=ndata

    

    with open('examples.json','w') as jfile:
        json.dump(exs,jfile,indent=4,sort_keys=True)


if __name__== "__main__":
    main()
