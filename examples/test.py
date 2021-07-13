import yaml

tempDictionary = {'n_particle': 1000000,
            'output': {'file': 'rad.uniform.out.txt', 'type': 'gpt'},
            'random_type': 'hammersley', 
            'start': {'MTE': {'units': 'meV', 'value': 150}, 'type': 'cathode'},
            'total_charge': {'units': 'pC', 'value': 10},
            'xy_dist': {'file': 'png.pdf.test.page' + 'x' + '.png', 'type': 'file2d',
                'min_x': {'value': -1, 'units': 'mm'},
                'max_x': {'value': 1, 'units': 'mm'},
                'min_y': {'value': -1, 'units': 'mm'},
                'max_y': {'value': 1, 'units': 'mm'},
                'threshold': 0.0}}
                    
with open(('pdf.test.page' + 'x' + '.in.yaml'),'w') as file:
    documents = yaml.dump(tempDictionary, file)