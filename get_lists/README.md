# GET LIST
`config.yaml`の内容通りにarxiv論文を引っ張ってくる

### subject
categoryを指定できる。<br>
```
subject: 'cat:cs.*'
```
`cat:cs.*`でcomputer science全般、<br>
さらにComputer Visionに絞るなどの場合は`subject: 'cat:cs.CV'`などとする。<br>
categoryはhttps://arxiv.org/category_taxonomy を参照。<br>

### keyword
検索したいkeywordを指定できる。<br>
```
keywords:
        agriculture: 7
        agritech: 7
        green house: 8
        environment sensors: 5
        remote sensing: 5
        database: 3
        machine learning: 3
```
個数の制限はない。<br>
また重みの合計値制約もなく、比率を表せばよい（後で標準化される）<br>


### num_papers
上位何件の論文リストを出力するかを決めることができる
```
num_papers: 100
```

### go_back_date
何日前まで遡るかを指定できる<br>
```
go_back_date: 365
```
`go_back_date=365`で直近1年分の検索となる。
