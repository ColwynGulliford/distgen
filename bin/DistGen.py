from optparse import OptionParser
from distgen.drivers import run_distgen

def main():

    """
    Main function for running distgen as a single execution program
    """

    #---------------------------------------------------------------------------------------
    # Parse Input Content and Run
    #---------------------------------------------------------------------------------------
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", default=None, 
                      help="write report to FILE", metavar="FILE")
    parser.add_option("-v",dest="verbose", default=0,help="Print short status messages to stdout")
    parser.add_option("-o",dest="output", default="gpt",help="Particle output type (default gpt)")
    parser.add_option("-p",dest="plots_on",default=False,action = "store_true")

    (options, args) = parser.parse_args()

    inputfile = options.filename
    verbose = int(options.verbose)  
    output = options.output    

    run_distgen(inputs=inputfile,output_type=output,verbose=verbose)
    #---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------- 
#   This allows the main function to be at the beginning of the file
# ---------------------------------------------------------------------------- 
if __name__ == '__main__':
    main()