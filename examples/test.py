from pdf2image import convert_from_path
import yaml

pdf = convert_from_path('examples/data/pdf.heatmap.pdf')

for q in range(len(pdf)):
    pdf[q].save('png.pdf.test.page' + str(q) + '.png','PNG')
    tempDictionary = [{'n_particle': 1000000, 
                       'output': {'file': 'rad.uniform.out.txt', 'type': 'gpt'},
                       'random_type': 'hammersley', 
                       'start': {'MTE': {'units': 'meV', 'value': 150}, 'type': 'cathode'},
                       'total_charge': {'units': 'pC', 'value': 10},
                       'xy_dist': {'file': 'png.pdf.test.page' + str(q) + '.png', 
                                   'max_x': {'units': 'mm', 'value': 1},
                                   'max_y': {'units': 'mm', 'value': 1},
                                   'min_x': {'units': 'mm', 'value': -1},
                                   'min_y': {'units': 'mm', 'value': -1},
                                   'threshold': 0.0,
                                   'type': 'file2d'}}]
    
    with open(('examples/data/' + 'pdf.test.page' + str(q) + '.in.yaml'),'w') as file:
        documents = yaml.dump(tempDictionary, file)