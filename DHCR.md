Đúng. Nếu dữ liệu đã **flatten conversation thành user–item / item–entity / item–word relations**, thì hướng “dynamic graph theo turn” là không hợp lý: ta không có (G_t) đáng tin cậy để học evolution. Mình rút lại hướng đó.

Sau khi nhìn lại kỹ lineage của 3 paper, mình nghĩ **novelty nên chuyển từ “graph evolves over time” sang “graph structure itself is semantically ambiguous and needs to be disentangled/learned.”**

Điểm mấu chốt là:

> **Các paper hiện tại rất giỏi tạo nhiều hypergraph, nhưng gần như mặc định rằng mỗi hyperedge được tạo ra đã là một preference relation đúng.**

HyCoRec lấy các item trong session làm hyperedge; entity/word hypergraph cũng được tạo từ các neighborhood bên ngoài.  HyFairCRS tiếp tục dùng item/entity/word/review-guided hypergraphs rồi tách mỗi hypergraph thành hypergraph + line graph và contrastive learning các representation. 

Vậy mình sẽ đi theo hướng khác hẳn:

# 1. Hướng mình đề xuất: **Hypergraph Disentanglement / Hyperedge Semantics**

Tên tạm:

> **Disentangled Hypergraph Interest Learning for Conversational Recommendation**

hoặc mạnh hơn:

> **Learning What Hyperedges Mean: Disentangled Multi-Interest Hypergraph Learning for Conversational Recommendation**

Core hypothesis:

[
\boxed{
\text{Observed hyperedge}
\neq
\text{single coherent user interest}
}
]

Một hyperedge hiện tại có thể chứa nhiều nguyên nhân khác nhau.

Ví dụ session có:

[
{Interstellar,\ Inception,\ Titanic,\ Avatar}.
]

HyCoRec có thể coi đây là một preference group. Nhưng thực tế có thể là:

[
\underbrace{{Interstellar,Inception,Avatar}}_{\text{sci-fi}}
]

và

[
\underbrace{{Titanic,Inception}}_{\text{romance}}
]

và thậm chí:

[
\underbrace{{Interstellar,Titanic}}_{\text{director/actor}}
]

Một hyperedge duy nhất đang **trộn nhiều latent relations**.

Đây là chỗ mình nghĩ có novelty tốt.

---

# 2. Vấn đề thực sự của HyCoRec / HiCore / HyFairCRS

Ba paper đều tăng richness bằng cách tăng số graph/view.

### HyCoRec

[
G^{item},G^{entity},G^{word}
]

và review.

### HiCore

[
{item,entity,word}
\times
{group,joint,purchase}.
]

### HyFairCRS

[
{item,entity,word,review}
]

rồi mỗi graph lại thành:

[
hypergraph + line\ graph.
]

HyFairCRS mô tả rất rõ pipeline này: tạo nhiều hypergraph từ historical conversation + external knowledge, sau đó dùng HGConv/GConv để tạo multiple interest representations và contrastive learning để refine chúng. 

**Nhưng tất cả đều có một assumption ngầm:**

[
\boxed{
\text{graph source}
\rightarrow
\text{meaningful relation}
}
]

Trong khi:

[
\text{same session}
]

không nhất thiết đồng nghĩa với:

[
\text{same interest}.
]

---

# 3. Novelty mới: không học "nhiều graph", mà học **nhiều semantics bên trong một graph**

Đây là khác biệt rất quan trọng.

Thay vì:

```text
item graph
entity graph
word graph
review graph
      ↓
4 embeddings
```

ta làm:

```text
                 observed hypergraph
                        │
                        ▼
              Hyperedge Disentangler
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Interest 1 Interest 2 Interest 3
             │          │          │
           Sci-fi     Romance     Actor
```

Tức:

[
H
\rightarrow
{H_1,H_2,\ldots,H_K}.
]

Đây là **hyperedge decomposition**.

---

# 4. Một hyperedge không còn là một relation

Giả sử hyperedge:

[
e={v_1,v_2,v_3,v_4,v_5}.
]

Thay vì:

[
e\rightarrow z_e
]

ta học:

[
e
\rightarrow
{z_e^1,z_e^2,\ldots,z_e^K}.
]

Mỗi (z_e^k) đại diện cho một latent semantic factor.

Ta có assignment:

[
A_{vk}
======

softmax_k
\left(
f(v,z_k)
\right).
]

Vậy incidence matrix ban đầu:

[
H\in{0,1}^{|V|\times |E|}
]

được factorize:

[
\boxed{
H
\approx
\sum_{k=1}^{K}
H^{(k)}
}
]

với:

[
H^{(k)}=H\odot A^{(k)}.
]

Đây là một **semantic hypergraph decomposition**.

---

# 5. Đây khác multi-interest learning hiện tại ở đâu?

Hiện tại các paper học:

[
X\rightarrow
[z_1,z_2,\ldots,z_K].
]

Nhưng (z_k) thường là latent vectors sau aggregation.

Ta đề xuất ngược lại:

[
\boxed{
z_k
\rightarrow
\text{explains which nodes/hyperedges}
}
]

Tức interest phải **có structural grounding**.

Ví dụ:

[
z_1
\rightarrow
{e_1,e_4,e_7}
]

[
z_2
\rightarrow
{e_2,e_5}
]

[
z_3
\rightarrow
{e_3,e_6,e_8}.
]

Như vậy ta có thể nói:

> Interest 1 không chỉ là một vector; nó là một sub-hypergraph.

Đây là điểm mình đánh giá mạnh.

---

# 6. Từ multi-interest thành **interest sub-hypergraphs**

Đây là formulation mình thích nhất:

[
\boxed{
\mathcal G_u
\rightarrow
{\mathcal G_u^1,\ldots,\mathcal G_u^K}
}
]

trong đó:

[
\mathcal G_u^k
==============

(V_u^k,E_u^k,H_u^k).
]

Mỗi interest:

[
I_u^k
\leftrightarrow
\mathcal G_u^k.
]

Do đó:

[
\text{interest}
\neq
\text{embedding slot}
]

mà:

[
\boxed{
\text{interest}
===============

\text{structured subgraph}
}
]

Đây là conceptual contribution khá đẹp.

---

# 7. Nhưng làm sao biết subgraph nào là "interest" nào?

Đây là phần learning objective.

Mình sẽ dùng **three constraints**.

## Constraint 1 — Cohesion

Các node trong cùng interest phải có semantic coherence:

[
\mathcal L_{coh}
================

-\sum_k
\frac{1}{|E_k|}
\sum_{e\in E_k}
sim(e,z_k).
]

Ví dụ:

[
{Star\ Wars,Alien,Interstellar}
]

nên coherent hơn:

[
{Star\ Wars,Titanic,The\ Notebook}.
]

---

# 8. Constraint 2 — Separation

Hai interest phải khác nhau:

[
\mathcal L_{sep}
================

\sum_{k\neq j}
\max
\left(
0,
sim(z_k,z_j)-m
\right).
]

Nhưng mình **không recommend chỉ dùng cosine orthogonality**.

Tốt hơn là separation ở **node/hyperedge assignment level**.

Ví dụ:

[
A_{ek}A_{ej}
]

nên nhỏ.

[
\mathcal L_{exclusive}
======================

\sum_e\sum_{k\neq j}
A_{ek}A_{ej}.
]

Nó buộc một hyperedge không bị mọi interest cùng "claim".

---

# 9. Constraint 3 — Coverage

Đây là thứ các multi-interest model thường thiếu.

Mỗi preference signal phải được explain bởi ít nhất một interest:

[
\sum_k A_{ek}\approx1.
]

Do đó:

[
\mathcal L_{cover}
==================

\sum_e
\left|
\sum_k A_{ek}-1
\right|.
]

Kết quả:

[
\boxed{
coherent
+
diverse
+
covering
}
]

thay vì chỉ:

[
\text{multiple vectors}.
]

---

# 10. Nhưng còn nhiều hypergraph views thì sao?

Đây là nơi có thể kế thừa HyFairCRS nhưng **đổi cách dùng**.

HyFairCRS cố align các view:

[
X_{item}
\leftrightarrow
X_{entity}
\leftrightarrow
X_{word}
\leftrightarrow
X_{review}.
]

Mình sẽ không align chúng hoàn toàn.

Thay vào đó:

[
\boxed{
\text{Cross-view correspondence}
}
]

Ví dụ:

```text
Item hypergraph
      │
      ├──── interest A
      │
Entity hypergraph
      │
      ├──── interest A
      │
Word hypergraph
      │
      └──── interest A
```

Tức **cùng một latent interest phải xuất hiện xuyên qua nhiều graph views**, nhưng mỗi view vẫn được phép giữ private information.

---

# 11. Shared–private interest decomposition

Cho view (v):

[
Z^{(v)}
=======

Z^{shared}
+
Z^{private,v}.
]

Ví dụ:

### Item view

[
z^{private,item}
]

capture item co-occurrence.

### Entity view

[
z^{private,entity}
]

capture actor/director/genre.

### Word view

[
z^{private,word}
]

capture semantic language.

Nhưng tất cả cùng có:

[
z^{shared}
==========

\text{underlying user interest}.
]

Đây là cách mình sẽ dùng contrastive learning.

---

# 12. Contrastive learning lúc này có ý nghĩa hơn HyFairCRS

Thay vì:

[
InfoNCE(X_i,X_e)
]

một cách global, ta contrast:

[
z_{u,k}^{item}
\leftrightarrow
z_{u,k}^{entity}.
]

Positive:

[
same\ user + same\ latent\ interest.
]

Negative:

[
same\ user + different\ interest.
]

Ví dụ:

[
z_{\text{sci-fi}}^{item}
]

positive với:

[
z_{\text{sci-fi}}^{entity},
]

nhưng negative với:

[
z_{\text{romance}}^{entity}.
]

Đây gọi là:

> **interest-aware cross-view contrastive learning.**

Nó có semantic interpretation rõ hơn pairwise view-level contrastive learning.

---

# 13. Một contribution rất mạnh: **Hyperedge-to-Interest Routing**

Mình sẽ biến nó thành module trung tâm.

Mỗi hyperedge:

[
e_j
]

routing tới K interests:

[
p(k|e_j,u)
==========

softmax
\left(
\frac{
q_k^T z_{e_j}
}{
\tau
}
\right).
]

Sau đó:

[
z_{u,k}
=======

\sum_j
p(k|e_j,u)
z_{e_j}.
]

Đây giống capsule/routing nhưng áp dụng ở **hyperedge level**.

Quan trọng:

> **Interest không được tạo trực tiếp từ user embedding; nó được hình thành bằng routing các hyperedges.**

Đây là khác biệt architecture rõ ràng.

---

# 14. Và có thể dùng line graph theo cách mới

HyFairCRS đã dùng line graph để model:

[
e_i\leftrightarrow e_j.
]

Ta không bỏ nó.

Nhưng thay đổi semantics:

### HyFairCRS

[
line\ graph
\rightarrow
hyperedge representation.
]

### Model mới

[
line\ graph
\rightarrow
hyperedge relation
\rightarrow
interest routing.
]

Tức line graph giúp phát hiện:

[
e_1,e_2,e_5
]

có structural coherence → có khả năng thuộc cùng một interest.

---

# 15. Thậm chí có thể định nghĩa **Interest Hypergraph**

Sau routing, tạo graph cấp cao:

[
\mathcal G_I
============

(\mathcal I,\mathcal E_I).
]

Trong đó:

[
\mathcal I
==========

{z_1,\ldots,z_K}
]

là interests.

Một hyperedge mới biểu diễn:

> các interests cùng xuất hiện trong một preference configuration.

Ví dụ:

[
{\text{sci-fi},\text{dark},\text{psychological}}.
]

Đây là **hypergraph-of-interests**.

Có hai level:

```text
Level 1
items/entities/words
        ↓
original hypergraph
        ↓
Level 2
latent interests
        ↓
interest hypergraph
```

Đây có thể trở thành một novelty rất đẹp:

[
\boxed{
\text{Hierarchical Hypergraph Representation}
}
]

---

# 16. Đây là hướng mình đánh giá cao hơn dynamic

Vì nó **không cần turn-level information**.

Input chỉ cần:

[
\mathcal H_u
]

được flatten từ historical conversation.

Ta vẫn có thể học:

[
\mathcal H_u
\rightarrow
{\mathcal H_u^1,\ldots,\mathcal H_u^K}.
]

Không cần:

[
\mathcal H_u^1
\rightarrow
\mathcal H_u^2
\rightarrow
\cdots.
]

Vì vậy hoàn toàn phù hợp với REDIAL/TG-REDIAL hiện tại.
