# Math

## 3D Rotation Matching With Least Squares Method - Derivation

It is known that

$$
\exists_2 q \in SU(2) \subset \mathbb{R}^4, \forall R \in SO(3) \subset \mathbb{R}^{3 \times 3}, \forall a \in \mathbb{R}^3, (0, R a) = q * (0, a) * q^{-1}
$$

where

$$
\forall (v_0, \vec{v}), (w_0,vec{w}) \in \mathbb{R}^4, (v_0, \vec{v}) * (w_0, \vec{w}) := (v_0 w_0 - \vec{v} \cdot \vec{w}, v_0 \vec{w} + w_0 \vec{v} + \vec{v} \times \vec{w})
$$

$$
(q_0, \vec{q})^{-1} = \frac{(q_0, -\vec{q})}{\|q\|^2}
$$

which shows that

$$
f_l, f_r: SU(2) \to SO(3) \text{ or } \mathbb{R}^{3} \to \mathbb{R}^{3 \times 3}, (q_0, q_1, q_2, q_3) \mapsto \begin{pmatrix} q_0 & -q_1 & -q_2 & -q_3 \\ q_1 & q_0 & -q_3 & q_2 \\ q_2 & q_3 & q_0 & -q_1 \\ q_3 & -q_2 & q_1 & q_0 \end{pmatrix}, \begin{pmatrix} q_0 & -q_1 & -q_2 & -q_3 \\ q_1 & q_0 & q_3 & -q_2 \\ q_2 & -q_3 & q_0 & q_1 \\ q_3 & q_2 & -q_1 & q_0 \end{pmatrix} \\
\forall p, q \in SU(2) \text{ or } \mathbb{R}^{3}, p * q = f_l(p) \begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{pmatrix} = f_r(q) \begin{pmatrix} p_0 \\ p_1 \\ p_2 \\ p_3 \end{pmatrix}
$$

Using this, as $\forall q \in \mathbb{R}^3, \|(0, \vec{q})\| = \|q\|$, we have

$$
\begin{aligned}
E(q) &:= \sum_k \|R a_k - b_k\|^2 \\
&= \sum_k \|q * a_k * q^{-1} - b_k\|^2 \\
&= \sum_k \|(q * a_k - b_k * q) * q^{-1}\|^2 \\
&= \sum_k \|q * a_k - b_k * q\|^2 \| q^{-1} \|^2 \\
&= \sum_k \|q * a_k - b_k * q\|^2 \\
&= \sum_k \|f_r(a_k) q - f_l(b_k) q\|^2 \\
&= \sum_k \|(f_r(a_k) - f_l(b_k)) q\|^2 \\
&= \sum_k q^T (f_r(a_k) - f_l(b_k))^T (f_r(a_k) - f_l(b_k)) q \\
&= q^T \left( \sum_k (f_r(a_k) - f_l(b_k))^T (f_r(a_k) - f_l(b_k)) \right) q \\
&= q^T B q
\end{aligned}
$$

which means $E(q)$ is a quadratic form.

As $B$ is symmetric, there exists an orthogonal matrix $P$ such that $B = P \Lambda P^{-1} = P \Lambda P^T$ where $\Lambda = \mathrm{diag} \{\lambda_i\} (\lambda_1 < \dots < \lambda_4)$ is diagonal. Let $r = P^T q$, then

$$
E(q) = q^T P B P^T q = (P^T q)^T D (P^T q) = r^T D r = \sum_i \lambda_i r_i^2
$$

under

$$
\| r \| = r^T r = q^T P^T P q = q^T q = 1
$$

and $E(q)$ is minimized when $r = (1, 0, 0, 0)^T$ and $q = P r = P (1, 0, 0, 0)^T = v_1$ where $v_1$ is the eigenvector corresponding to $\lambda_1$.

## References

- [3D shape matching with quaternions \| lisyarus blog](https://lisyarus.github.io/blog/posts/3d-shape-matching-with-quaternions.html)
- [Showing the Correctness of Quaternion Rotation](https://erkaman.github.io/posts/quaternion_rotation.html#mjx-eqn-eqquatprod)
- [クォータニオン計算便利ノート](https://www.mesw.co.jp/business/report/pdf/mss_18_07.pdf)
- [ゲームプログラマのための数学の歩き方 - クォータニオンとリー群編](http://www.jp.square-enix.com/tech/library/pdf/CEDEC2021_SQEX_IMI_Quaternion_20210827_public.pdf)
