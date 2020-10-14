---
layout: post
title: "Association Rule Mining"
date: 2020-06-10 11:00:24
image: associationrule.png
tags: blog
usemathjax: true

---

<b>Association rule mining</b> is the basis for pattern recognition within multiple applications to find frequent relations, causal structures or correlations between occurrences of data. Common applications of association rule mining are market basket analysis, fraud detection, medical diagnosis, cross-marketing etc. Association guides decision making based on the historical data that exists. 

An association rule is represented in the format:
<ul>
<li>Antecedent -> Consequent, </li>
</ul>
which are defined by the measures of interestingness i.e support and confidence following a given threshold. These established rules can be be categorised as: <b>actionable rules</b>- having relevant insights, <b>trivial rules</b>- known to people with domain knowledge and <b>inexplicable</b>- relationships have no explainable insight and impact.
There a couple of measures used with regard to the association rules. 

<h3>The measures of interestingness</h3>

<b>Support</b> is the frequency of the rule within the data. It is given by:
<ul>
<li>support(A -> B) = p(A and B) </li>
</ul>

<b>Confidence</b> is the percentage of occurrences containing A which also contain B. Given by:
<ul>
<li>confidence(A -> B) = p(B|A) = sup(A,B)/sup(A) </li>
</ul>


<h3>Apriori algorithm</h3>
This algorithm relates to associative rule mining from frequent occurrence sets by relying on the downward closure property.

“The Apriori algorithm generates candidates with smaller length k first and counts their supports before generating
candidates of length $$(k+1)$$. The resulting frequent k-itemsets are used to restrict the number of $$(k + 1)$$-candidates
with the downward closure property”

This phenomenon restricts the cost for generating frequent patterns or occurrences based on the support threshold defined.

![RF]({{ site.baseurl }}/images/apriori.PNG){: style="display:block; margin: 0 auto; width:520px;height:267px"}
There are other frequent itemset mining algorithms like enumeration tree, recursive growth suffix based pattern growth

<h3>Anti monotone property</h3>
To implement the anti monotone property while generating the association rules; the following theorem is held on the confidence measure;

"Let $$Y$$ be an itemset and $$X$$ is a subset of $$Y$$ . If a rule $$X \rightarrow Y - X $$ does not satisfy the confidence threshold, then any rule 
$$\tilde{X} \rightarrow Y -\tilde{X}$$,where $$\tilde{X}$$ is a subset of $$X$$, must not satisfy the confidence threshold as well."

This theorem can be proved by comparing the confidence of the rules $$X \rightarrow Y - X $$  and $$\tilde{X} \rightarrow Y -\tilde{X}$$. The confidence of the rules are sup($$Y$$)/sup($$X$$)  and sup($$Y$$)/sup($$\tilde{X}$$). $$\tilde{X}$$ being a subset of $$X$$, then sup($$\tilde{X}$$) $$\geq$$ sup($$X$$), therefore the rule $$\tilde{X} \rightarrow Y -\tilde{X}$$ can not have a higher confidence than $$X \rightarrow Y - X $$.This is applicable for a given itemset.

This implies that if a certain rule of a given itemset is discarded that it does not meet the minimum confidence threshold, then all the subsets of the antecedent of the particular rule do not need to be explored as they will definitely have a lower confidence. This increases the speed of the program as not all combinations of itemsets do not have to be explored. 

<h4>Resources </h4>

<ul>
    <li>Data Mining by Charu C. Aggarwal</li>
</ul>

<h4> Code implementation in Matlab </h4>
{% highlight matlab %}

% input parameters: minsup = minimum support, minconf = minimum confidence,
% antimonotone = true(to use the property)/ false(ignore the property)
function rules = associationRules(minsup,minconf, antimonotone)
    shoppingList = readDataFile;

    ntrans = size(shoppingList,1);
    items = unique([shoppingList{:}]);
    nitems = numel(items);

    [tridx,trlbl] = grp2idx(items);

    % Create the binary matrix
    dataset = zeros(ntrans,nitems);
    for i = 1:ntrans
       dataset(i,tridx(ismember(items,shoppingList{i}))) = 1;
    end

    % Generate frequent items of length 1
    support{1} = sum(dataset)/ntrans;
    f = find(support{1} >= minsup);
    frequentItems{1} = tridx(f);
    support{1} = support{1}(f);
    % Generate frequent item sets
    k = 1;
    while k < nitems && size(frequentItems{k},1) > 1
        % Generate length (k+1) candidate itemsets from length k frequent itemsets
        frequentItems{k+1} = [];
        support{k+1} = [];

        % Consider joining possible pairs of item sets
        for i = 1:size(frequentItems{k},1)-1
            for j = i+1:size(frequentItems{k},1)
                if k == 1 || isequal(frequentItems{k}(i,1:end-1),frequentItems{k}(j,1:end-1))
                    candidateFrequentItem = union(frequentItems{k}(i,:),frequentItems{k}(j,:));  
                    if all(ismember(nchoosek(candidateFrequentItem,k),frequentItems{k},'rows'))                
                        sup = sum(all(dataset(:,candidateFrequentItem),2))/ntrans;                    
                        if sup >= minsup
                            frequentItems{k+1}(end+1,:) = candidateFrequentItem;
                            support{k+1}(end+1) = sup;
                        end
                    end
                else
                    break;
                end            
            end
        end         
        k = k + 1;
    end

    % antimonotone implementation
    rules = {}; % cell array of extracted rules
    ct = 1; % count of the rule generated
    for i =2:length(support)-1 % no association rules for the  first itemset
        Lset = frequentItems{i};
        for itemset=1:size(Lset,1)
            % specific frequent itemset
            whichset = Lset(itemset,:); 

            allcombin ={}; % all subsets in this itemset
            for j=1:(length(whichset)-1) % get combinations of size k-1
                allsets = nchoosek(whichset,j); %subsets S of whichset of size j
                for row=1:size(allsets,1)
                    allcombin{end+1,1} = allsets(row,:);
                end
            end

            allcombin = flip(allcombin);
            disantecedents = {}; % discarded antecedents for anitmonotone   
            %for every non empty subset s of x output the rule S => I-S

            for f=1:length(allcombin)
                if antimonotone
                    %anti-monotone
                    % If a rule S->(I ?S) does not satisfy the confidence threshold, 
                    % then any rule S?-> (I ? S?),where S? is a subset of S,
                    % must not satisfy the confidence threshold as well

                    %check if combination is a subset of any of the discarded
                    occur = cellfun(@ismember,repmat(allcombin(f), size(disantecedents)), disantecedents, 'UniformOutput',false);
                    found = any(cellfun(@all, occur)); % true or false if any of the combinations is a subset of the 

                    if found
                        %if allcombination is a subset, then skip to next iteration
                        continue
                    end
                end
                notsubset = setdiff(whichset, allcombin{f}); %I-S

                %extracting the support values
                whichsupcol = length(allcombin{f}); % cell index dependent on size of itemset
                indexincol = find(ismember(frequentItems{whichsupcol}, allcombin{f},'rows'));
                conf = support{i}(itemset)/support{whichsupcol}(indexincol);
                if antimonotone && conf < minconf
                    disantecedents{end+1,1} = allcombin{f};
                else if conf >= minconf
                    rules{ct, 1} = items(allcombin{f});
                    rules{ct, 2} = items(notsubset);
                    rules{ct, 3} = conf;
                    ct = ct+1;
                end

            end


        end
    end

end
{% endhighlight %}

