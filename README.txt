Install/Setup Steps:
1. Download and Extract https://gtvault-my.sharepoint.com/:u:/r/personal/apierce35_gatech_edu/Documents/GT_Shared/ML/Project1/apierce35_ML_P1.zip?csf=1&e=1VDifg

2. Make sure scikit-learn, numpy, pandas, matplotlib, and all required dependencies are installed
with Python 3.7+
    pip3 install scikit-learn, numpy, pandas, matplotlib

3. All code is executed from experiments.py, which is executed as 'python3 experiments.py'

4. experiments.py is written to be executed manually multiple times in succession, manipulating the
constants at the top of the main function in order to do the necessary experiments.

    a. algorithm constant defines which algorithm you are executing. Available options are
        'dt', 'nn', 'svm', 'ab', or 'knn'
    b. dataset constant defines which dataset you are using. Avilable options are
        'adult', 'baby'
    c. test constant defines what test you are doing. Available optinos are
        'gridseach', 'learningcurve', 'modelcomplexity', 'purning', 'testdata'
    d. stepNum constant defines what step in the tuning process you are on. Every execution of
    experiments.py will append results to a txt file titled [DatasetName]_[Algorithm].txt.
    Additionally, most tests you can run will generate a .png image of a results graph. This stepNum
    constant helps keep the order you did your operations in straight when looking back through
    results, and the graphs in chronological order when sorting files by name.
    e. updateTxt constant defines if that execution will write to a file or not - useful if debugging
    new code.
    
5. Seperate files were produced for every dataset (all prefixed by data_*.py) and each learner (all
prefixed by learner_*.py). These contain the necessary configuration for tests (like ranges for model
complexity analysis, or filenames, or titles). To customize these parameters, please see the
respective file.

Other Information
2. All raw data files are in the data/ folder, including many files not used in the report. I have
kept them in for completeness, as well as their associated data_*.py folder should the reader wish to
try some of these out. They are all already setup with the required pre-processing steps to have any
learner_*.py be able to learn on them.

2. All results used in the report are also contained in the zip, under the plots/ folder. In this
directory, there will be 5 other directories, by learner type. In each of those, the appropriate
results .txt files and graphs (.png) will be found for Adult and Baby. You may find quite a few more
graphs than included in the report, as was mentioned. Many were omitted due to space limitations
and/or dead-ends in the tuning process.