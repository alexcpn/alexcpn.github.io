---
layout: post
title:  "Welcome to My Site!"
---

{% include_relative test.liquid %}

# Welcome

**Hello world**, this is my second Jekyll blog post 

I hope you like it!

using  include; The branch.liquid should be in _inlcudes folder

{{ raw }}
{% include branch.liquid %}
{{ endraw }}


This is the current branch {{ branchName }}
