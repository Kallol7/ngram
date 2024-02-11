# MIT License

# Copyright (c) 2024 Kallol7

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np

class NGram:
    """ An optized way to create n-gram only using numpy
    
    Input: 
        "A brown fox"
    
    Output: 
        [array(['A'], dtype='<U5'), array(['brown'], dtype='<U5'), 
        array(['fox'], dtype='<U5'), array(['A', 'brown'], dtype='<U5'), 
        array(['brown', 'fox'], dtype='<U5'), array(['A', 'brown', 'fox'], dtype='<U5')]
    
    Example:    
        ng = NGram("A brown fox") 
        ngrams_from_1_to_3 = ng.ngram(3)
        only_bi_gram = ng.ngram(2)
        only_tri_gram = ng.ngram(3)
    """
    
    def __init__(self,text):
        self.words = np.array(text.split())
        self._n_gram_opt = np.frompyfunc(self._return_n_gram,1,1)
    
    def _return_n_gram(self,idx):
        return self.words[idx:idx+self.n]

    def specific_ngram(self,n):
        assert n<=len(self.words), f"n should be less then or equal to the total number of tokens. Total tokens = {len(self.words)}"
        self.n = n
        
        indices = np.arange(len(self.words)-(n-1))
        return self._n_gram_opt(indices)
    
    def ngram(self, a, b = None):
        if b is None:
            b=a+1
            a=1
        else:
            b=b+1
                
        out = []
        for i in range(a,b):
            out.extend(self.specific_ngram(i))
        return out
    

if __name__=="__main__":
    demo = NGram(text="A brown fox ").ngram(3)

    print('\nInput: \n  A brown fox"\n')
    print(f"Output:\n  {demo}\n")
    
    multiplier = 500000
    text = "A long text which is very long " * multiplier
    
    print(f"\nBig text length: {len(text)} characters")
    
    # Total 3500000 tokens
    print(f"Total {len(text.split())} tokens after splitting the text\n") 
    
    
    # initialiaze the object
    ng = NGram(text)
    
    # 14 Million, for n = 1,2,3,4
    print(
        int(len(ng.ngram(4))/1e6), 
        "Million n-grams, for n = 1,2,3,4\n"
    )

    # 3.5 Million, for n = 1
    print(
        len(ng.specific_ngram(1))/1e6,
        "Million, for n = 1"
    )

    # 3.499999 Million, for n = 2
    print(
        len(ng.specific_ngram(2))/1e6,
        "Million, for n = 2"
    )

    # 3.499998 Million, for n = 3
    print(
        len(ng.specific_ngram(3))/1e6,
        "Million, for n = 3"
    )

    # 3.499997 Million, for n = 4
    print(
        len(ng.specific_ngram(4))/1e6,
        "Million, for n = 4"
    )
