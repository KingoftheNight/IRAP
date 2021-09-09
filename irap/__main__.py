import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(file_path)
try:
    from . import Visual
    from . import Version
except:
    import Visual
    import Version

def irap():
    print('\nEasyRAAC version=' + Version.version)
    Visual.visual_create_blast(file_path)
    Visual.visual_create_aaindex(file_path)
    Visual.visual_create_raac(file_path)

# main
if __name__ == '__main__':
    irap()
