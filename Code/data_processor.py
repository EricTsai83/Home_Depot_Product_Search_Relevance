# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
@important note: 
    the script is altered by ChenglongChen 3rd place solution for HomeDepot product search 
    results relevance competition on Kaggle.
@author: Eric Tsai <eric492718@gmail.com>
@brief: process data
        - a bunck of processing
        - automated spelling correction
        - query expansion
        - extract product name for search_term and product_title

"""

# +
# basic libraries
import numpy as np
import pandas as pd

# utils
import csv  
from importlib.machinery import SourceFileLoader
import multiprocessing
from bs4 import BeautifulSoup  # 處理 html tag
from collections import Counter

# NLP
import nltk  # 一套基於 Python 的自然語言處理工具箱
import regex  # it is necessary because using the re module will get an error

# libraries we write
import config
from utils import ngram_utils, pkl_utils, logging_utils, time_utils
from spelling_checker import GoogleQuerySpellingChecker, AutoSpellingChecker

#--------------------------- Processor ---------------------------
## base class
## Most of the processings can be casted into the "pattern-replace" framework
class BaseReplacer:
    '''
    re.sub(pattern,repl,string,count)
    pattern: pattern which we want to replace. It write by regular expression.
    repl:replacer
    string: the string we need to deal with
    count:number of match pattern that we want to replace. If I choose 0, it means replacing all the targets.
    '''
    def __init__(self, pattern_replace_pair_list=[]):
        self.pattern_replace_pair_list = pattern_replace_pair_list
    def transform(self, text):
        for pattern, replace in self.pattern_replace_pair_list:
            try:
                text = regex.sub(pattern, replace, text)
            except ValueError:
                print(ValueError)
        return regex.sub(r'\s+', ' ', text).strip() # \s+ : means "one or more spaces" replace to one space


def test_BaseReplacer():
    text = 'Eric  Tsai Tsai Tsai Tsai is good'
    pattern_replace_pair_list = [('Tsai', 'Eric'), ('good', 'bad')]
    assert BaseReplacer(pattern_replace_pair_list).transform(text)=='Eric Eric Eric Eric Eric is bad'

class LowerCaseConverter(BaseReplacer):
    """
    Traditional -> traditional
    """
    # Method Overriding (方法覆寫)，覆寫從 BaseReplacer 繼承的方法
    def transform(self, text):
        return text.lower()


def test_LowerCaseConverter():
    text = 'EricTsai Tsai Tsai Tsai is good'
    assert LowerCaseConverter().transform(text) == 'erictsai tsai tsai tsai is good'


# + tags=["script"]
class LowerUpperCaseSplitter(BaseReplacer):
    """
    homeBASICS Traditional Real Wood -> homeBASICS Traditional Real Wood

    hidden from viewDurable rich finishLimited lifetime warrantyEncapsulated panels ->
    hidden from view Durable rich finish limited lifetime warranty Encapsulated panels

    Dickies quality has been built into every product.Excellent visibilityDurable ->
    Dickies quality has been built into every product Excellent visibility Durable

    BAD CASE:
    shadeMature height: 36 in. - 48 in.Mature width
    minutesCovers up to 120 sq. ft.Cleans up
    PUT one UnitConverter before LowerUpperCaseSplitter

    Reference:
    https://www.kaggle.com/c/home-depot-product-search-relevance/forums/t/18472/typos-in-the-product-descriptions
    """
    def __init__(self):
        ########################################################################
        # The first regular expression can solve the problem which is a 
        # sentence connect with the other sentence but without any character.
        ########################################################################
        # The second regular expression means: a Blank characters, followed 
        # by the low case letter, followed by a upper case letter character.
        self.pattern_replace_pair_list = [(r'(\w)[\.?!]([A-Z])', r'\1 \2'), # \1: means the group 1 element. In this case, it is items which match with pattern (\w)  
                                          (r'(?<=( ))([a-z]+)([A-Z]+)', r'\2 \3'),]  # \2: means the group 2 element



def test_LowerUpperCaseSplitter():
    assert (LowerUpperCaseSplitter().transform('homeBASICS Traditional Real Wood')
            == 'homeBASICS Traditional Real Wood')
    assert (LowerUpperCaseSplitter().transform('hidden from viewDurable rich finishLimited lifetime warrantyEncapsulated panels') 
            == 'hidden from view Durable rich finish Limited lifetime warranty Encapsulated panels')
    assert (LowerUpperCaseSplitter().transform('shadeMature height: 36 in. - 48 in.Mature width')
            == 'shadeMature height: 36 in. - 48 in Mature width')
    assert (LowerUpperCaseSplitter().transform(' shadeMature height: 36 in. - 48 in.Mature width')
           =='shade Mature height: 36 in. - 48 in Mature width')


'''
Create word replacement patterns, using homemade replacement words.
Input will be a CSV file that contains one column and each value is
a text which shows the words, followed by ",", and followed by a word we
want to replace. Note, texts is annotated if it begins with '#' character.
'''
class WordReplacer(BaseReplacer):
    '''
    if words are not near the [a-z0-9_](\W or ^\w), 
    replce it by replacement dictionary
    '''
    def __init__(self, replace_fname):
        self.replace_fname = replace_fname # file name which is alread create in config
        self.pattern_replace_pair_list = []
        for line in csv.reader(open(self.replace_fname)): # use csv.reader will return a list which seperate by ","
            if len(line) == 1 or line[0].startswith('#'):
                continue # The continue statement is used to skip the rest of the code inside a loop for the current iteration only. Loop does not terminate but continues on with the next iteration. 
            try: # Regular Expression means: a text is between two characters which are non-alphanumeric characters 
                pattern = r'(?<=\W|^)%s(?=\W|$)'%line[0] # or a text is first character in the string (^)
                replace = line[1]                        # or a text is the end of the string ($)
                self.pattern_replace_pair_list.append( (pattern, replace) )
            except:
                print(line)
                pass

            

def test_WordReplacer():
    replace_list = WordReplacer(replace_fname=config.WORD_REPLACER_DATA).pattern_replace_pair_list
    assert regex.sub(replace_list[0][0], replace_list[0][1],'Eric Tsai want a undercabinet') == 'Eric Tsai want a under cabinet'
    assert regex.sub(replace_list[0][0], replace_list[0][1],'Eric Tsai want a 2undercabinet') == 'Eric Tsai want a 2undercabinet'
    assert regex.sub(replace_list[0][0], replace_list[0][1],'Eric Tsai want a $undercabinet') == 'Eric Tsai want a $under cabinet'
    assert regex.sub(replace_list[0][0], replace_list[0][1],'undercabinet is what you need') == 'under cabinet is what you need'
    assert regex.sub(replace_list[0][0], replace_list[0][1],'The undercabinet is what you need') == 'The under cabinet is what you need'
    assert regex.sub(replace_list[0][0], replace_list[0][1],'*undercabinet is what you need') == '*under cabinet is what you need'
    assert regex.sub(replace_list[0][0], replace_list[0][1],'undercabinet% is what you need') == 'under cabinet% is what you need'
    assert regex.sub(replace_list[0][0], replace_list[0][1],'undercabinet_ is what you need') == 'undercabinet_ is what you need'


# <code style="background:yellow;color:black">***The class below has a bug in Chenglong version. The pattern was wrong. But I already took care of it. The original version can't deal with many '-' characters in the text like 'Vinyl-Leather-Rubber'.***</code>

## deal with letters
class LetterLetterSplitter(BaseReplacer):
    """
    For letter and letter
    /:
    Cleaner/Conditioner -> Cleaner Conditioner

    -:
    Vinyl-Leather-Rubber -> Vinyl Leather Rubber

    For digit and digit, we keep it as we will generate some features via math operations,
    such as approximate height/width/area etc.
    /:
    3/4 -> 3/4

    -:
    1-1/4 -> 1-1/4
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r'(?<=[a-zA-Z])[/\-](?=[a-zA-Z])', r' ')
        ]



def test_LetterLetterSplitter():
    assert LetterLetterSplitter().transform('Cleaner/Conditioner') == 'Cleaner Conditioner'
    assert LetterLetterSplitter().transform('Vinyl-Leather-Rubber') == 'Vinyl Leather Rubber'
    assert LetterLetterSplitter().transform('Vinyl-Leather/Rubber') == 'Vinyl Leather Rubber'
    assert LetterLetterSplitter().transform('COVID-19 is crazy') == 'COVID-19 is crazy'
    assert LetterLetterSplitter().transform('3/4') == '3/4'
    assert LetterLetterSplitter().transform('1-1/4') == '1-1/4'


## deal with digits and numbers
class DigitLetterSplitter(BaseReplacer):
    """
    x:
    1x1x1x1x1 -> 1 x 1 x 1 x 1 x 1
    19.875x31.5x1 -> 19.875 x 31.5 x 1

    -:
    1-Gang -> 1 Gang
    48-Light -> 48 Light

    .:
    includes a tile flange to further simplify installation.60 in. L x 36 in. W x 20 in. ->
    includes a tile flange to further simplify installation. 60 in. L x 36 in. W x 20 in.
    """
    
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r'(\d+)[\.\-]*([a-zA-Z]+)', r'\1 \2'),
            (r'([a-zA-Z]+)[\.\-]*(\d+)', r'\1 \2'),
        ]



def test_DigitLetterSplitter():
    assert DigitLetterSplitter().transform('1.a') == '1 a'
    assert DigitLetterSplitter().transform('1x') == '1 x'
    assert DigitLetterSplitter().transform('1x1x1x1x1') == '1 x 1 x 1 x 1 x 1'
    assert DigitLetterSplitter().transform('1-Gang') == '1 Gang'
    assert DigitLetterSplitter().transform('COVID-19 is crazy') == 'COVID 19 is crazy'
    assert DigitLetterSplitter().transform('3/4') == '3/4'
    assert DigitLetterSplitter().transform('1-1/4') == '1-1/4'


class DigitCommaDigitMerger(BaseReplacer):
    """
    1,000,000 -> 1000000
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(?<=\d+),(?=000)", r""),
        ]


def test_DigitCommaDigitMerger():
    assert DigitCommaDigitMerger().transform('1,000,000') == '1000000'
    assert DigitCommaDigitMerger().transform('900,000') == '900000'
    assert DigitCommaDigitMerger().transform('80,000') == '80000'


class NumberDigitMapper(BaseReplacer):
    """
    one -> 1
    two -> 2
    """
    def __init__(self):
        numbers = [
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
            'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'
        ]
        digits = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90
        ]
        self.pattern_replace_pair_list = [
            (r'(?<=\W|^)%s(?=\W|$)'%n, str(d)) for n,d in zip(numbers, digits)
        ]



def test_NumberDigitMapper():
    assert NumberDigitMapper().transform('one') == '1'
    assert NumberDigitMapper().transform('ten') == '10'
    assert NumberDigitMapper().transform('fifty') == '50'
    

## deal with unit
class UnitConverter(BaseReplacer):
    """
    shadeMature height: 36 in. - 48 in.Mature width
    PUT one UnitConverter before LowerUpperCaseSplitter
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([0-9]+)( *)(inches|inch|in|in\.|')(?=[^\w]+)\.?", r"\1 in. "),
            (r"([0-9]+)( *)(pounds|pound|lbs|lb|lb\.)(?=[^\w]+)\.?", r"\1 lb. "),
            (r"([0-9]+)( *)(foot|feet|ft|ft\.|'')(?=[^\w]+)\.?", r"\1 ft. "),
            (r"([0-9]+)( *)(square|sq|sq\.) ?\.?(inches|inch|in|in.|')(?=[^\w]+)\.?", r"\1 sq.in. "),
            (r"([0-9]+)( *)(square|sq|sq\.) ?\.?(feet|foot|ft|ft.|'')(?=[^\w]+)\.?", r"\1 sq.ft. "),
            (r"([0-9]+)( *)(cubic|cu|cu\.) ?\.?(inches|inch|in|in.|')(?=[^\w]+)\.?", r"\1 cu.in. "),
            (r"([0-9]+)( *)(cubic|cu|cu\.) ?\.?(feet|foot|ft|ft.|'')(?=[^\w]+)\.?", r"\1 cu.ft. "),
            (r"([0-9]+)( *)(gallons|gallon|gal)(?=[^\w]+)\.?", r"\1 gal. "),
            (r"([0-9]+)( *)(ounces|ounce|oz)(?=[^\w]+)\.?", r"\1 oz. "),
            (r"([0-9]+)( *)(centimeters|cm)(?=[^\w]+)\.?", r"\1 cm. "),
            (r"([0-9]+)( *)(milimeters|mm)(?=[^\w]+)\.?", r"\1 mm. "),
            (r"([0-9]+)( *)(minutes|minute)(?=[^\w]+)\.??", r"\1 min. "),
            (r"([0-9]+)( *)(°|degrees|degree)(?=[^\w]+)\.?", r"\1 deg. "),
            (r"([0-9]+)( *)(v|volts|volt)(?=[^\w]+)(?=[^\w]+)\.?", r"\1 volt. "),
            (r"([0-9]+)( *)(wattage|watts|watt)(?=[^\w]+)\.?", r"\1 watt. "),
            (r"([0-9]+)( *)(amperes|ampere|amps|amp)(?=[^\w]+)\.?", r"\1 amp. "),
            (r"([0-9]+)( *)(qquart|quart)(?=[^\w]+)\.?", r"\1 qt. "),
            (r"([0-9]+)( *)(hours|hour|hrs\.)(?=[^\w]+)\.?", r"\1 hr "),
            (r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)(?=[^\w]+)\.?", r"\1 gal. per min. "),
            (r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)(?=[^\w]+)\.?", r"\1 gal. per hr "),
        ]


def test_UnitConverter():
    assert UnitConverter().transform('shadeMature height: 36 in. - 48 in.Mature width') == 'shadeMature height: 36 in. - 48 in. Mature width'
    assert UnitConverter().transform('shadeMature height: 36 in - 48 in.Mature width') == 'shadeMature height: 36 in. - 48 in. Mature width'
    assert UnitConverter().transform('shadeMature height: 36 inch - 48 in.Mature width') == 'shadeMature height: 36 in. - 48 in. Mature width'
    assert UnitConverter().transform('shadeMature height: 36in. - 48in.Mature width') == 'shadeMature height: 36 in. - 48 in. Mature width'
    assert UnitConverter().transform('shadeMature height: 36in - 48in Mature width') == 'shadeMature height: 36 in. - 48 in. Mature width'
    assert UnitConverter().transform('shadeMature height: 36inyy - 48in Mature width') == 'shadeMature height: 36inyy - 48 in. Mature width'
    assert UnitConverter().transform("shadeMature height: 36' - 48in Mature width") == 'shadeMature height: 36 in. - 48 in. Mature width'
    assert UnitConverter().transform('Top 100 instagram Business Accounts sorted by Followers') == 'Top 100 instagram Business Accounts sorted by Followers'
    assert UnitConverter().transform('100 inches is long') == '100 in. is long'
    assert UnitConverter().transform('100 inches is 50 cmiii uu') == '100 in. is 50 cmiii uu'
    assert UnitConverter().transform('100 vampire is cool') == '100 vampire is cool'



## deal with html tags
class HtmlCleaner:
    def __init__(self, parser):
        self.parser = parser  # 'html.parser'

    def transform(self, text):
        bs = BeautifulSoup(text, self.parser)
        text = bs.get_text(separator=' ')
        return text


def test_HtmlCleaner():
    text = '''<p>Hi. This is a simple example.<br>Yet poweful one.<p><a href="http://example.com/">I linked to <i>example.com</i></a>'''
    assert HtmlCleaner(parser='html.parser').transform(text) == 'Hi. This is a simple example. Yet poweful one. I linked to  example.com'


## deal with some special characters
# 3rd solution in CrowdFlower (Create by Chenglong)
class QuartetCleaner(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r'<.+?>', r''),  # <at least one characters)
            # html codes（character entities）
            (r'&nbsp;', r' '),  # Non-Breakable Space
            (r'&amp;', r'&'),  # &amp; is the character reference for "An ampersand".
            (r'&#39;', r"'"),
            (r'/>/Agt/>', r''),
            (r'</a<gt/', r''),
            (r'gt/>', r''),
            (r'/>', r''),
            (r'<br', r''),
            # do not remove ['.', '/', '-', '%'] as they are useful in numbers, e.g., 1.97, 1-1/2, 10%, etc.
            (r'[&<>)(_,;:!?\+^~@#\$]+', r' '),  # in the Chenglong, will remove space, but in this case no need
            ("'s\\b", r''),
            (r"[']+", r''),
            (r'[\"]+', r''),
        ]



def test_QuartetCleaner():
    '''
    unit test for specific tasks and whole function
    '''
    # specific task
    text = '<remove me> Hello my friends!<a>'
    assert regex.sub(r'<.+?>', r'', text) == ' Hello my friends!'
    text = 'Hello&nbsp;my friends!<a>'
    assert regex.sub(r'&nbsp;', r' ', text) == 'Hello my friends!<a>'
    text = 'Hello my friends &amp; mentor'
    assert regex.sub(r'&amp;', r'&', text) == 'Hello my friends & mentor'
    text = 'I&#39;m a man'
    assert regex.sub(r'&#39;', r"'", text) == "I'm a man"
    text = '/>/Agt/> must be remove'
    assert regex.sub(r'/>/Agt/>', r'', text) == ' must be remove'
    text = '</a<gt/ must be remove'
    assert regex.sub(r'</a<gt/', r'', text) == ' must be remove'
    text = 'gt/> must be remove'
    assert regex.sub(r'gt/>', r'', text) == ' must be remove'
    text = '/> must be remove'
    assert regex.sub(r'/>', r'', text) == ' must be remove'
    text = '<br must be remove'
    assert regex.sub(r'<br', r'', text) == ' must be remove'
    text = '&<>)(_,;:!?+^~@#$ must be remove'
    assert regex.sub(r'[&<>)(_,;:!?\+^~@#\$]+', r'', text) == ' must be remove'
    text = "'s\\b must be remove"
    assert regex.sub(r"'s\\b", r'', text) == ' must be remove'
    text = "I'm Eric"
    assert regex.sub(r"[']+", r'', text) == 'Im Eric'
    text = '"Eric Tsai" is my name.'
    assert regex.sub(r'[\"]+', r'', text) == 'Eric Tsai is my name.'
    # whole function
    text = '<remove me> Hello my friends!<a>'
    assert QuartetCleaner().transform(text) == 'Hello my friends'
    
    

## lemmatizing for using pretrained word2vec model
# 2nd solution in CrowdFlower
class Lemmatizer:
    '''
    can delete white space and \n(new line character)
    nltk.stem.wordnet.WordNetLemmatizer().lemmatize(token):
        You need to manually specify the part of speech(pos). 
        If you don't set the pos parameter that the default is noun, so only plural nouns can be converged here.
    '''
    def __init__(self):
        self.Tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.Lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    def transform(self, text):
        tokens = [self.Lemmatizer.lemmatize(token) for token in self.Tokenizer.tokenize(text)]  #cut word and do Lemmatisation
        return ' '.join(tokens) # 'a b c'


    
def test_Lemmatizer():
    text = '''I found my computers yesterday.
              It took me along\n time. 
              You guys know, a data scientist\n has many computers which is a normal thing.
              By the way, I worked out to fit my body yesterday.
              I hope is, are, and been can transform to be.
              dishes cities  knives beliefs heroes volcanoes'''
    assert Lemmatizer().transform(text) == 'I found my computer yesterday. It took me along time. You guy know , a data scientist ha many computer which is a normal thing. By the way , I worked out to fit my body yesterday. I hope is , are , and been can transform to be. dish city knife belief hero volcano'


    
## stemming
class Stemmer:
    '''
    Convert uppercase to lowercase
    Delete common endings of words
    Can't delete white space and \n(new line character)
    Can't deal with Lemmatization. Ex. working => work
    In generally, snowball is better than porter. So snowball method will be default.
    '''
    def __init__(self, stemmer_type='snowball'):
        self.stemmer_type = stemmer_type
        if self.stemmer_type == 'porter':
            self.stemmer = nltk.stem.PorterStemmer()
        elif self.stemmer_type == 'snowball':
            self.stemmer = nltk.stem.SnowballStemmer('english')

    def transform(self, text):
        tokens = [self.stemmer.stem(token) for token in text.split(" ")]
        return ' '.join(tokens)


def test_Stemmer():
    text = '''Good muffin cost $ 3.88 in New York. Please buy me two of them. Thanks . 
          I worked from home last \n year when COVID-19 disaster.
          I am working.
          '''
    Stemmer().transform(text) == 'good muffin cost $ 3.88 in new york. pleas buy me two of them. thank . \n          i work from home last \n year when covid-19 disaster.\n          i am working.\n          '



class ProcessorWrapper:
    '''
    help function input convert to string
    '''
    def __init__(self, processor):
        self.processor = processor

    def transform(self, input):
        if isinstance(input, str): # check input whether is an str class instance
            out = self.processor.transform(input)
        elif isinstance(input, float) or isinstance(input, int):
            out = self.processor.transform(str(input))
        elif isinstance(input, list):
            # take care when the input is a list
            # currently for a list of attributes
            out = [0]*len(input)
            for i in range(len(input)):
                out[i] = ProcessorWrapper(self.processor).transform(input[i])
        else:
            raise(ValueError(f'Currently not support type: {type(input).__name__}'))
        return out


def test_ProcessorWrapper():
    s='lolololoooooo'
    assert ProcessorWrapper(processor=Stemmer()).transform(s) == 'lolololoooooo'
    assert ProcessorWrapper(processor=Stemmer()).transform(3.14159265358) == '3.14159265358'
    assert ProcessorWrapper(processor=Stemmer()).transform(['a','b','b','c', 4, 5, 6.589]) == ['a', 'b', 'b', 'c', '4', '5', '6.589']


class ListProcessor:
    """
    WARNING: This class will operate on the original input list itself
    """
    def __init__(self, processors):
        self.processors = processors

    def process(self, lst):
        for i in range(len(lst)):
            for processor in self.processors:
                lst[i] = ProcessorWrapper(processor).transform(lst[i])
        return lst



def test_ListProcessor():
    processors = [Lemmatizer(), Stemmer()]
    lst = ['Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.', 'Eric Tsai want a under cabinet.']
    assert ListProcessor(processors).process(lst) == ['good muffin cost $ 3.88 in new york. pleas buy me two of them. thank .',
 'eric tsai want a under cabinet .']


class DataFrameProcessor:
    """
    WARNING: This class will operate on the original input dataframe itself
    """
    def __init__(self, processors):
        self.processors = processors

    def process(self, series):  # I change the variable name: df-->series, because it is more clearly.
        for processor in self.processors:
            series = series.apply(ProcessorWrapper(processor).transform)
        return series


def test_DataFrameProcessor():
    processors = [Lemmatizer(), Stemmer()]
    dic = {'name': ['Eric', 'Ben'], 
           'age': ['27', '58'],
           'education': ['National Chengchi University', 'Northwestern University']}
    df = pd.DataFrame(data = dic)
    for i in range(len(df.name)):
        assert DataFrameProcessor(processors).process(df.name)[i] == pd.Series(['eric', 'ben'], name='name')[i]

        
class DataFrameParallelProcessor:
    """
    WARNING: This class will operate on the original input dataframe itself

    https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
    """
    def __init__(self, processors, n_jobs=1):  # my notebook only has 2 CPU
        self.processors = processors
        self.n_jobs = n_jobs

    def process(self, dfAll, columns):
        df_processor = DataFrameProcessor(self.processors)
        p = multiprocessing.Pool(self.n_jobs)
        dfs = p.imap(df_processor.process, [dfAll[col] for col in columns])
        for col,df in zip(columns, dfs):
            dfAll[col] = df
        return dfAll

#------------------- Query Expansion -------------------
# 3rd solution in CrowdFlower (Chenglong team decided to remove the feature which might be a major cause of overfitting.)
class QueryExpansion():
    '''
    if stopwords_threshold decrease, the number of stop word will raise
    '''
    def __init__(self, df, ngram=3, stopwords_threshold=0.9, base_stopwords=set()):  # base_stopwords: can add some stop word at the start
        self.df = df[["search_term", "product_title"]].copy()
        self.ngram = ngram
        self.stopwords_threshold = stopwords_threshold
        self.stopwords = set(base_stopwords).union(self._get_customized_stopwords())
        
    def _get_customized_stopwords(self):
        '''
        get stopwords with low frequency
        '''
        words = " ".join(list(self.df["product_title"].values)).split(" ")
        # find word frequency
        counter = Counter(words)
        # find unique word
        num_uniq = len(list(counter.keys()))
        # define the amount of stop word (if a word frequency is too high, then it will be a stopword)
        num_stop = int((1.-self.stopwords_threshold)*num_uniq)
        stopwords = set()
        for e,(w,c) in enumerate(sorted(counter.items(), key=lambda x: x[1])):
            if e == num_stop:
                break
            stopwords.add(w)
        return stopwords

    def _ngram(self, text):
        tokens = text.split(" ")
        tokens = [token for token in tokens if token not in self.stopwords]
        return ngram_utils._ngrams(tokens, self.ngram, " ")

    def _get_alternative_query(self, lst):
        ''' 
        Through product describtion to find the most frequency word in ngram list 
        '''
        res = []
        for v in lst:
            res += v
        c = Counter(res)
        value, count = c.most_common()[0]
        return value

    def build(self):
        '''
        Through the above function '_get_alternative_query' to set the data process
        '''
        self.df["title_ngram"] = self.df["product_title"].apply(self._ngram)
        corpus = self.df.groupby("search_term").apply(lambda x: self._get_alternative_query(x["title_ngram"]))
        corpus = corpus.reset_index()
        corpus.columns = ["search_term", "search_term_alt"]
        self.df = pd.merge(self.df, corpus, on="search_term", how="left")
        return self.df["search_term_alt"].values


def test_QueryExpansion():
    # create data frame
    dic = {'search_term': ['Eric', 'Ben', 'Eric'], 
       'age': ['27', '58', '66'],
       'product_title': ['a a a a b b b b c c c c d d d d e e e e f f f f g g g g h h h h i i i i j j j j k k k k l l l l mostA mostA mostA mostA mostA mostA',
                         'a a a a b b b b c c c c d d d d e e e e f f f f g g g g h h h h i i i i j j j j k k k k l l l l mostB mostB mostB mostB mostB mostB',
                         'stopword_1 stopword_2 stopword_3 stopword_4 stopword_5']}
    df_1 = pd.DataFrame(data = dic)
    assert QueryExpansion(df_1)._get_customized_stopwords() == {'stopword_1'}
    
    assert QueryExpansion(df_1)._ngram('a b c d e f g h') == ['a b c', 'b c d', 'c d e', 'd e f', 'e f g', 'f g h']
    
    '''
    assume text = aaaaabc
    n=3
    ngram_list = ['a a a', 'a a a', 'a a a', 'a a b', 'a b c']
    '''
    # function: '_get_alternative_query' will group by search term, so I set the search term be the same here
    dic = {'search_term': ['Eric', 'Eric', 'Eric'], 
           'product_title': ['aaaaabc',
                             'bbbbbac',
                             'aaaaabc']}
    df_2 = pd.DataFrame(data = dic)
    # we can find out the compound word(or string) 'a a a' has the most frequency here
    ngram_series = pd.Series([['a a a', 'a a a', 'a a a', 'a a b', 'a b c'],
                              ['b b b', 'b b b', 'b b b', 'b b a', 'b a c'],
                              ['a a a', 'a a a', 'a a a', 'a a b', 'a b c']])
    # so excepted output is 'a a a'
    assert QueryExpansion(df_2)._get_alternative_query(ngram_series) == 'a a a'
    
    assert (QueryExpansion(df_1).build()[0] 
            == np.array(['mostA mostA mostA', 'mostB mostB mostB', 'mostA mostA mostA'], dtype='object'))[0]
    
    assert (QueryExpansion(df_1).build()[1]
            == np.array(['mostA mostA mostA', 'mostB mostB mostB', 'mostA mostA mostA'], dtype='object'))[1]
    
    assert (QueryExpansion(df_1).build()[2]
            == np.array(['mostA mostA mostA', 'mostB mostB mostB', 'mostA mostA mostA'], dtype='object'))[2]


# ### Extract Product Name
# 1. Find color string, using homemade pattern.
# 2. Find units string, using UnitConverter(), I create before.
# 3. Find the other pattern, which is not product name and remove it.

#------------------- Extract Product Name -------------------
# 3rd solution in CrowdFlower
color_data = SourceFileLoader('COLOR_LIST', config.COLOR_DATA).load_module()  # import specific .py file which is not has same path with this file
COLORS_PATTERN = r'(?<=\W|^)%s(?=\W|$)'%('|'.join(color_data.COLOR_LIST))
UNITS = [' '.join(r.strip().split(' ')[1:]) for p,r in UnitConverter().pattern_replace_pair_list]
UNITS_PATTERN = r'(?:\d+[?:.,]?\d*)(?: %s\.*)?'%('|'.join(UNITS))
DIM_PATTERN_NxNxN = r'%s ?x %s ?x %s'%(UNITS_PATTERN, UNITS_PATTERN, UNITS_PATTERN)
DIM_PATTERN_NxN = r'%s ?x %s'%(UNITS_PATTERN, UNITS_PATTERN)



# 3rd solution in CrowdFlower
class ProductNameExtractor(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            # Remove descriptions (text between paranthesis'()'/brackets'[]')
            (r'[ ]?[[(].+?[])]', r''),
            # Remove 'made in...'
            (r'made in [a-z]+\b', r''),
            # Remove descriptions (hyphen'-' or comma',' followed by space then at most 2 words, repeated)
            (r'([,-]( ([a-zA-Z0-9]+\b)){1,2}[ ]?){1,}$', r''),
            # Remove descriptions (prepositions staring with: with, for, by, in )
            (r'\b(with|for|by|in|w/) .+$', r''),
            # colors & sizes
            (r'size: .+$', r''),
            (r'size [0-9]+[.]?[0-9]+\b', r''),
            (COLORS_PATTERN, r''),
            # dimensions
            (DIM_PATTERN_NxNxN, r''),
            (DIM_PATTERN_NxN, r''),
            # measurement units
            (UNITS_PATTERN, r''),
            # others
            (r'(value bundle|warranty|brand new|excellent condition|one size|new in box|authentic|as is)', r''),
            # stop words
            (r'\b(in)\b', r''),
            # hyphenated words
            (r'([a-zA-Z])-([a-zA-Z])', r'\1\2'),
            # special characters
            (r'[ &<>)(_,.;:!?/+#*-]+', r' '),
            # numbers that are not part of a word
            (r'\b[0-9]+\b', r''),
        ]
        
    def preprocess(self, text):
        pattern_replace_pair_list = [
            # Remove single & double apostrophes
            (r'[\']+', r''),
            # Remove product codes (long words (>5 characters) that are all caps, numbers or mix pf both)
            # don't use raw string format
            (r'[ ]?\b[0-9A-Z-]{5,}\b', r''),
        ]
        text = BaseReplacer(pattern_replace_pair_list).transform(text)
        text = LowerCaseConverter().transform(text)
        text = DigitLetterSplitter().transform(text)
        text = UnitConverter().transform(text)
        text = DigitCommaDigitMerger().transform(text)
        text = NumberDigitMapper().transform(text)
        text = UnitConverter().transform(text)
        return text
        
    def transform(self, text):
        text = super().transform(self.preprocess(text))
        text = Lemmatizer().transform(text)
        text = Stemmer(stemmer_type='snowball').transform(text)
        # last two words in product
        text = ' '.join(text.split(' ')[-2:])
        return text

#------------------- Process Attributes -------------------
def _split_attr_to_text(text):
    attrs = text.split(config.ATTR_SEPARATOR)  # ' | '
    return ' '.join(attrs)

def _split_attr_to_list(text):
    attrs = text.split(config.ATTR_SEPARATOR)        
    if len(attrs) == 1:
        # missing
        return [[attrs[0], attrs[0]]]
    else:  # attrs[::2]: means return values which are according to indexes order 0,2,4,6,... 
        return [[n,v] for n,v in zip(attrs[::2], attrs[1::2])]  # attrs[1::2]: means return values which are according to indexes order 1,3,5,... 

def test_split_attr_to_text():
    text = 'A | A_attr | B | B_attr | C | C_attr'
    assert _split_attr_to_text(text) == 'A A_attr B B_attr C C_attr'


def test_split_attr_to_list():
    text = 'A | A_attr | B | B_attr | C | C_attr'
    assert _split_attr_to_list(text) == [['A', 'A_attr'], ['B', 'B_attr'], ['C', 'C_attr']]


    
# ## Data Processing Modular Design
def main():
    ### 1. Record Time
    now = time_utils._timestamp()
    ###########
    ## Setup ##
    ###########
    logname = f'data_processor_{now}.log'
    logger = logging_utils._get_logger(config.LOG_DIR, logname)


    # Put product_attribute_list, product_attribute and product_description first as they are
    # quite time consuming to process.
    # Choose the columns by check data_preparer.ipynb. In the end, the notebook will show the clean data frame.
    columns_to_proc = [
        # # product_attribute_list is very time consuming to process
        # # so we just process product_attribute which is of the form 
        # # attr_name1 | attr_value1 | attr_name2 | attr_value2 | ...
        # # and split it into a list afterwards
        # 'product_attribute_list',
        'product_attribute_concat',
        'product_description',
        'product_brand', 
        'product_color',
        'product_title',
        'search_term', 
    ]
    if config.PLATFORM == 'Linux':
        config.DATA_PROCESSOR_N_JOBS = len(columns_to_proc)

    # clean using a list of processors
    processors = [
        LowerCaseConverter(), 
        # See LowerUpperCaseSplitter and UnitConverter for why we put UnitConverter here
        # 其實沒差，除非能處理掉數字加介係詞 in 的狀況不被替代成單位 in.(inch)
        UnitConverter(),
        LowerUpperCaseSplitter(), 
        WordReplacer(replace_fname=config.WORD_REPLACER_DATA), 
        LetterLetterSplitter(),
        DigitLetterSplitter(), 
        DigitCommaDigitMerger(), 
        NumberDigitMapper(),
        UnitConverter(), 
        QuartetCleaner(), 
        HtmlCleaner(parser='html.parser'), 
        Lemmatizer(),
    ]
    stemmers = [
        Stemmer(stemmer_type='snowball'), 
        Stemmer(stemmer_type='porter')
    ][0:1]  # means only use Stemmer(stemmer_type='snowball')

    ## simple test
    text = '1/2 inch rubber lep tips Bullet07'
    print('Original:')
    print(text)
    list_processor = ListProcessor(processors)
    print('After:')
    print(list_processor.process([text]))

    #############
    ## Process ##
    #############
    ## load raw data
    dfAll = pkl_utils._load(config.ALL_DATA_RAW)
    columns_to_proc = [col for col in columns_to_proc if col in dfAll.columns]

    if config.TASK == 'sample':
        dfAll = dfAll.iloc[0:config.SAMPLE_SIZE]
        print(f'data length: {len(dfAll)}')

    ## extract product name from search_term and product_title
    ext = ProductNameExtractor()
    dfAll['search_term_product_name'] = dfAll['search_term'].apply(ext.transform)
    dfAll['product_title_product_name'] = dfAll['product_title'].apply(ext.transform)
    if config.TASK == 'sample':
        print(dfAll[['search_term', 'search_term_product_name', 'product_title_product_name']])

    ## clean using GoogleQuerySpellingChecker(Chenglong team not used in final submission)
    # MUST BE IN FRONT OF ALL THE PROCESSING
    if config.GOOGLE_CORRECTING_QUERY:
        logger.info('Run GoogleQuerySpellingChecker at search_term')
        checker = GoogleQuerySpellingChecker()
        dfAll['search_term'] = dfAll['search_term'].apply(checker.correct)

    ## clean uisng a list of processors
    df_processor = DataFrameParallelProcessor(processors, config.DATA_PROCESSOR_N_JOBS)
    df_processor.process(dfAll, columns_to_proc)
    # split product_attribute_concat into product_attribute and product_attribute_list
    dfAll['product_attribute'] = dfAll['product_attribute_concat'].apply(_split_attr_to_text)
    dfAll['product_attribute_list'] = dfAll['product_attribute_concat'].apply(_split_attr_to_list)
    if config.TASK == 'sample':
        print(dfAll[['product_attribute', 'product_attribute_list']])


    # query expansion (Chenglong team decided to remove the feature which might be a major cause of overfitting.)
    if config.QUERY_EXPANSION:
        list_processor = ListProcessor(processors)
        # stop words must to access data process. EX. NumberDigitMapper function will replace 'one' to '1'.
        # So, if stop word has 'one', it must replace to '1',too. 
        base_stopwords = set(list_processor.process(list(config.STOP_WORDS)))  # a set of stop word
        qe = QueryExpansion(dfAll, ngram=3, stopwords_threshold=0.9, base_stopwords=base_stopwords)
        dfAll['search_term_alt'] = qe.build()
        if config.TASK == 'sample':
            print(dfAll[['search_term', 'search_term_alt']])

    # save data
    logger.info(f'Save to {config.ALL_DATA_LEMMATIZED}')
    columns_to_save = [col for col in dfAll.columns if col != 'product_attribute_concat']
    pkl_utils._save(config.ALL_DATA_LEMMATIZED, dfAll[columns_to_save])


    ## auto correcting query(Chenglong team not used in final submission)
    if config.AUTO_CORRECTING_QUERY:
        logger.info('Run AutoSpellingChecker at search_term')
        checker = AutoSpellingChecker(dfAll, exclude_stopwords=False, min_len=4)
        dfAll['search_term_auto_corrected'] = list(dfAll['search_term'].apply(checker.correct))
        columns_to_proc += ['search_term_auto_corrected']
        if config.TASK == 'sample':
            print(dfAll[['search_term', 'search_term_auto_corrected']])
        # save query_correction_map and spelling checker
        fname = '%s/auto_spelling_checker_query_correction_map_%s.log'%(config.LOG_DIR, now)
        checker.save_query_correction_map(fname)
        # save data
        logger.info('Save to %s'%config.ALL_DATA_LEMMATIZED)
        columns_to_save = [col for col in dfAll.columns if col != 'product_attribute_concat']
        pkl_utils._save(config.ALL_DATA_LEMMATIZED, dfAll[columns_to_save])

    ## clean using stemmers
    df_processor = DataFrameParallelProcessor(stemmers, config.DATA_PROCESSOR_N_JOBS)
    df_processor.process(dfAll, columns_to_proc)
    # split product_attribute_concat into product_attribute and product_attribute_list
    dfAll['product_attribute'] = dfAll['product_attribute_concat'].apply(_split_attr_to_text)
    dfAll['product_attribute_list'] = dfAll['product_attribute_concat'].apply(_split_attr_to_list)


    # query expansion
    if config.QUERY_EXPANSION:
        list_processor = ListProcessor(stemmers)
        base_stopwords = set(list_processor.process(list(config.STOP_WORDS)))
        qe = QueryExpansion(dfAll, ngram=3, stopwords_threshold=0.9, base_stopwords=base_stopwords)
        dfAll['search_term_alt'] = qe.build()
        if config.TASK == 'sample':
            print(dfAll[['search_term', 'search_term_alt']])

    # save data
    logger.info('Save to %s'%config.ALL_DATA_LEMMATIZED_STEMMED)
    columns_to_save = [col for col in dfAll.columns if col != 'product_attribute_concat']
    pkl_utils._save(config.ALL_DATA_LEMMATIZED_STEMMED, dfAll[columns_to_save])

      
if __name__ == "__main__":
    main()