# An optimized way to create n-gram only using numpy.

## Input:

    "A brown fox"

## Output: 
    
    [array(['A'], dtype='<U5'), array(['brown'], dtype='<U5'), 
    array(['fox'], dtype='<U5'), array(['A', 'brown'], dtype='<U5'), 
    array(['brown', 'fox'], dtype='<U5'), array(['A', 'brown', 'fox'], dtype='<U5')]

## Example:    
    
    ng = NGram("A brown fox") 
    ngrams_from_1_to_3 = ng.ngram(3)
    only_bi_gram = ng.ngram(2)
    only_tri_gram = ng.ngram(3)
