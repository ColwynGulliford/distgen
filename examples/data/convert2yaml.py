import yaml
import os

def main():

    for filename in os.listdir(os.getcwd()):
        if(filename.endswith('json')):
            with open(filename) as fid:
                data = yaml.safe_load(fid)

            newfile = '.'.join( (filename.split('.'))[:-1] )+'.yaml'
            with open(newfile,'w') as yid:
                yaml.dump(data, yid, default_flow_style=False)


if __name__==main():
    main()
