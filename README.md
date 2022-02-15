Jekyll and Liquid based Github.io supported site. Using MathJax 

Converting to PDF 

alexcpn.github.io/html/NN/ml$ pandoc -s -o 3_gradient_descent.pdf 3_gradient_descent.md  --pdf-engine=xelatex --variable urlcolor=cyan

Testing locally
alexcpn.github.io$ sudo apt-get install ruby-full
alexcpn.github.io$ sudo gem install bundler
alexcpn.github.io$ bundle update
alexcpn.github.io$ bundle install
alexcpn.github.io$ bundle exec jekyll serve

