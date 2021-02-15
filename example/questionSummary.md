# lyft
> - taxi service, car maintainance, debit card, a bridge between driver and passenger
> - Core values:
>     + Be yourself
>     + Uplift others
>     + Make it happen - Own the work. Focus on impact.
> - Open to CA, NYC and Seattle

# Must read
> - [1point3arces summary](https://www.1point3acres.com/bbs/interview/lyft-software-engineer-309636.html)
> - [Coding questions](https://docs.google.com/spreadsheets/d/1h4CAkjg2TMvchitPYWOZ03o_hnK8C5TYKt1bcxCuICU/edit#gid=0)
> - [Lyft blog](https://eng.lyft.com/)

## Behavioral question
### Why lyft
### Most challenging project?
### How to deal with a people without ownership

## HR interview
> - 刚和hr 小姐姐聊了一下，粉车车有8个vertical team, 感觉分的很细

## Tel interview
### [Level 5 link](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=532045&highlight=%B7%DB%B3%B5)
> - 
面的是level 5 team，一个工作一年多的小哥面的。题目是首先定义Linked list node，然后写个函数删除第n个node。Follow up是detect cycl以及return cycle length。

### 127

## 716 

### 642 May be

### [Fix bugs](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=395284&highlight=level%2B5)
```c++
连连看 >>>> 让你肉眼纠错，我了个去，比如肉眼找unquire point， uint 非负值的错之类的。 醉了。 好多东西因为coding style的关系不是很常用，所以一下子想不起来，做了差不多5-7道题吧，挂了。
```

### [LC238](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=544784&highlight=lyft)

### [link](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=544215&highlight=lyft)
> - Determine the minimum number of steps it takes to get from 1 to a target number N using the following operations:-Multiply by 2
-Divide by 3 (truncate decimals)

### 3. 白板coding，就是在一个图上找最短路径的问题，刚开始是无权连通图，后来是有权连通图。基本就是bfs然后dijkstra搞定了
```
题目是有一个 paginated API 参数是 pageId， 每次返回该pageid的elements，现实fetchN.

e.g. 3 pages, [ [0,1,2,3,4], [5,6,7,8,9], [10,11,12] ]
paginated(0) = [0,1,2,3,4],

paginated(1) =  [5,6,7,8,9],


fetchN(2) = [0,1]
fetchN(2) = [2,3]
fetchN(2) = [4,5]
fetchN(100) = [6,7,8,9,10,11,12]
```

### [LC157](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=510629&highlight=lyft)

### 小行星撞击 LC375
```
美国小哥问了高频的行星碰撞 (利口其三无)
不过和利口的不完全一样。。。
思路也是用个栈撸一下
```

### LC735

### LC42

### [LC333](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=536087&highlight=lyft)

### LC347

### LC279

### LC200. Number of islands
> - follow up area of islands

### [questions](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=469423&highlight=%B7%DB%B3%B5)

### [similiar to LC128, LC146](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=416203&highlight=%B7%DB%B3%B5)
```
题目应该不难，但是楼主这个转专业狗实在是....有愧于论坛，无法抬头。

一个array, [8，-2, -1，-10,3,4,5,6,7,9,10，0，1，2]. 找到最长的consecutive sequence，注意不是返回长度，而是数组本身，which means -2 -1 0 1 2 3 4 5 6 7 8 9 .撸主longest ascending subsequence 题有点忘，但是这里的难点是，array并不是有序的。一开始说直接sort 然后linear scan, 小哥说行啊，但是要求on方法。然后我就想，卧槽无序怎么搞呢。

我左思右想没想起来ascending subsequence的具体，最后瞎想了一个code没有过一些corner case...已跪。不冤，刷题程度不够，没什么可吐槽的。大家以后共勉。
```

### LC283
> - 蠡口  耳捌散  （O(n), O(1))两个方法最后被问了公司每天的data volume

### [add-binary](https://www.lintcode.com/problem/add-binary/description)

### LC238. Product of Array Except Self

### LC68. text justification
> - Read and write from file.

### [Similiarto LC158](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=533594&highlight=%B7%DB%B3%B5)
```c++
Given an API to fetch page (10 results on each page):

MAX_RESULTS = 103

def fetch_page(page):
    page = page or 0 # start from page 0 if not specified
    return {
        "page": page + 1,
        "results": [range(10 * page, min(10 * (page + 1), MAX_RESULTS))]
    }

Implement a ResultRecher class with a fetch method to return required number of results:
class  ResultRecher:

    def fetch_page(num_results):

和利口 要五八(read4) 一个思路。Follow up:
1. what if the fetch_page has a 30% possibility to fail? -> add retry, implement retry with a limit
2. what if sometimes the server may be overloaded, retry immediately will make it even worse? no need to consider auto scale -> add wait time between retries
what if sometimes the server may be overloaded, retry immediately will make it even worse? -------
这个可以用 exponential retry 的做法. aws 的 sdk 都是这么实现的 https://docs.aws.amazon.com/general/latest/gr/api-retries.html
```

##  detect cycle in linked list
面的是level 5 team，一个工作一年多的小哥面的。题目是首先定义Linked list node，然后写个函数删除第n个node。Follow up是detect cycle以及return cycle length

##
```
先问Given two arrays of integers, print elements that appear in both. 口头说算法和复杂度，并没让写。

让写的是一个CommonIterator class, constructor的输入是两个iterators that return sorted integers, 主要是写hasNext() 和 next(). 可以用它们print the intersection of two arrays in sorted order。比如

i1 = iter([1, 2, 3, 4, 5, 6])
i2 = iter([2, 3, 6, 7])

  while CI.hasNext():
   print CI.next()

Output:
iter = CommonIterator(i1, i2)
iter.next() -> 2
iter.next() -> 3
iter.next() -> 6


   求大米看面经。

```
## [LC300](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=495465&highlight=%B7%DB%B3%B5)
```
蠡口伞凌翎，经典题，直接用python的bisect_left算法六行秒了。之后要求打印序列，先写了一个在dp数组里存path的算法。然后面试官要求优化空间，用类似最短路径的方法，对每个node存prev就成。但是写的时候脑子又姜化了，几个辅助数组乱飞差点没写出来，还好前面省下了大量时间还是磨出来了。最后Q&A结束。
```

## [link](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=486902&highlight=%B7%DB%B3%B5)
```
N个string合并成一个，要保留原来string里面字符的顺序。求所有可能组合
merge(['AB', 'C']) => ['ABC', 'ACB', 'CAB']
merge(['AB', 'CD']) => ['ABCD', 'ACBD', 'ACDB', 'CABD', 'CADB', 'CDAB']

merge(['AB', 'CD', 'E']) => ['ABCDE', 'ABCED', 'ABECD', 'ACBDE', 'ACBED', 'ACDBE', 'ACDEB', 'ACEBD', 'ACEDB', 'AEBCD', 'AECBD', 'AECDB', 'CABDE', 'CABED', 'CADBE', 'CADEB', 'CAEBD', 'CAEDB', 'CDABE', 'CDAEB', 'CDEAB', 'CEABD', 'CEADB', 'CEDAB', 'EABCD', 'EACBD', 'EACDB', 'ECABD', 'ECADB', 'ECDAB']

## [link](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=482179&highlight=%B7%DB%B3%B5)
```
给定一个类SingleStreamReader 该类有一个方法Read 一次能最多一个数据流中最多读出n个字节（n是输入参数） 实现一个类MultiStreamReader 实现一下三个方法
1: AddSteam 加入一个Stream
2. RemoveStream 删除一个Stream
3. Read 从数据流中读取最多n个字节

如果一个数据流读完类 按照加入顺序读取下一个. From 1point 3acres bbs
还挺有意思的题 考察基本数据结构，接口设计，集成和编程能力
// [Detailed problem definition](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=506641&highlight=%B7%DB%B3%B5)
题目是给一个stream class，有一个method read(n)，返回值是一个list，从这个流里读取n个item，如果没有n个就返回余下的所有item。问如何实现一个multistream class，支持以下方法：1. add一个stream object2. remove 一个stream object（和面试官clarify了，暂时不考虑读完stream以后或者读到一半时call remove该stream的情况）3. read(n), 从当前的所有stream objects里面按add的顺序读n个item，如果没有n个，返回余下的所有item。主要难点是read。解法一：用一个queue保存added streams，初始化已读的item count=0，初始化返回的list=[]。当queue里还有stream时循环，pop队头的stream，call stream.read(n - count)，数一下读出了多少个item，把读到的count加到total count里，把读到的item加到返回的list里。如果total count >= n就跳出循环。但这个解法的缺点是，remove方法的时间复杂度是O(N)，空间复杂度也是O(N)，因为new了一个list去存没被remove的stream objects。解法二：用Python的OrderedDict，把stream object当做key（和面试官确认了假设这个stream class有哈希method），可以做到O(1)的add和remove，并且能像queue一样从队头pop。面试官问为什么是O（1），解释了一下这个data structure相当于一个dictionary加一个双向链表，通过key找到相应的链表节点，因为是双向的，所以能O（1）remove，然后把key从dictionary里面删除，dictionary的pop（key）是O（1）Follow-up：多线程的情况下，add和read是否需要同步。我的回答是取决于business requirement，如果想读尽可能多，那么需要同步（其实感觉没必要同步，加了新stream read可以照样读到，除非不想读）。但是tricky的是remove和read，如果读到一半remove，那就mess up了，所以remove和read需要同步。
```

## [link](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=435356&highlight=%B7%DB%B3%B5)
```
先是两轮电话面试 第一轮的题目是leetcode小行星碰撞 面试官很注重写test case
第二轮不太常规 不是考题目 是面试官在codepad上面不停贴代码让改错和优化 要先改才能run 包括warning也会要求改掉 内容不难 主要是一些小bug +1 -1数组超过边界什么的 考点涉及到 stl 容器的基本操作 冒泡排序。。我当时忘得差不多 写了很久 。。
onsite 第一轮pubsub 的设计 面试官给了register(signalid，calkbacks)unregister(signalid，calkbacks) call(signalid)三个接口 让写这三个函数
第二轮是算法 考了两个数组找公共元素 重复和不重复都要考虑 让写了用set的方法。然后又考了permutation，写了dp的方法。都很详细地问了时间和空间复杂度。
然后是lunch和manager聊 问了一些简历
最后一轮是lc trap rain water 我这一题之前靠背的 。。。交流的时候不太顺利 虽然写出来了。。
```


补充内容 (2019-2-26 14:27):
用recursive + cache 解决。估计用iterative DP也行
```

## [link](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=172124&highlight=lyft)
```

刚面完，妥妥的挂了。。。。白人小哥
一道题: 直接上来给了一串string log，说 parse the log，然后小哥就静音干自己的事去了。。。
每一行［e.g］[02/01/2014 5:7:8 + 0000] PUT /user/4324/riders/543534 HTTP1.1 304 chrome ...
要求输出:
(1) 统计每一个request出现的次数 （request type + request url mapping  + status code 组合）
(2) unique user
写完之后，说不对。。。user个数不对
仔细看log, 只有跟在user后面的才算并且不是每个log都有user information。
改完后说，输出request统计输出不对，应该把data全部用'#'表示。
改完后说，还是不对，要求request的输出按照count个数排序。
。。。。然，并没有时间改了。。。。。
攒个经验，下次写题目之前要把要求问清楚了。
回去继续好好联系基本功
```

## Onsite
### [1point3arces](https://www.1point3acres.com/bbs/thread-515161-1-1.html)

### [link](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=535792&highlight=lyft)
```
onsite:
第一个：带时间版本的key-value系统，久巴一，用自己的电脑做
第二个：经验 / behavior
第三个：给两个iterator指向sorted array，实现一个iterator给下一个在两个数组里面的共同元素
第四个：设计，lyft的coupon code系统
```

### [Good summary of onsite questions till Apr 29th 2019](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=519882&highlight=%B7%DB%B3%B5)

### [Onsite from leetcode](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=515161&highlight=%B7%DB%B3%B5)
```
coding: 中国小姐姐，n叉树最大路径（weight都在edge上）。follow up: bfs和dfs的区别和优缺点？（复杂度都是O(n)，但是树太深dfs用recursion写会爆栈）如果weight可能为负数哪个更好？如果需要找到k top weight path需要怎么变？
system design: 印度大姐 + 中国小哥shadow，design twitter。
system design: 印度小哥，design donation website，我提出需要用message queue实现donation submission与payment的decouple，之后主要在讨论需要传递哪些消息，如何handle各模块的timeout/failure
bar raiser: 白人大哥，是无人车下面的一个team lead，介绍the most proud project和behavior questions。这轮答得不好，花了太多时间介绍背景，被面试官打断要求直接说为什么觉得proud……当时没被Amazon洗礼过，面对experience/behavior question不知道该怎么回答。后面几家面试说多了以后才逐渐有了感觉。

```


## System design
### Distributed web crawler
```
2. System design: 设计distributed web crawler, 给你1000台slave，需要定期parse wiki，怎么样能及时update any changes in wiki articles. 这个记得先算一下qps和需要的storage，我当时忘了算task table需要的容量，设计了个sharded sql db, 结果后来在提示下算了一下这个storage的容量，发现只需要500MB，所以只需要一个内存的小storage就可以了。。。
```

### Donation system
```
Design - Donation system，重点在于怎么处理high qps，以及和很flaky的3rd party credit card processor对接，怎么防止重复扣款之类。我参考了Uber payment系统来答：https://www.youtube.com/watch?v=5TD8m7w1xE0。
```

