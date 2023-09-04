# Formulation

## High-Temperature Series Expansion

Let $\mathbf{H}(\{\mathbf{r}_{i}\}, \{\mathbf{s}_{i}\})$ be the total energy of given atomic and spin configuration.

\begin{align*}
    Z = \sum_{\{\mathbf{s}_{i}\}} \exp(-\beta \,\mathbf{H}(\{\mathbf{r}_{i}\}, \{\mathbf{s}_{i}\}))
\end{align*}

We expand the partition function with the Taylor series around $\beta = 0$.

```{math}
:label: eq_1st_expansion
    Z &= \sum_{\{\mathbf{s}_{i}\}} \left( 1 - \beta \,\mathbf{H}(\{\mathbf{r}_{i}\}, \{\mathbf{s}_{i}\}) +
        \frac{\beta^{2} \,\mathbf{H}^{2}(\{\mathbf{r}_{i}\}, \{\mathbf{s}_{i}\})}{2!} + \cdots +
        \frac{(- \beta)^{n}\,\mathbf{H}^{n}(\{\mathbf{r}_{i}\}, \{\mathbf{s}_{i}\})}{n!} + \cdots + \right) \\
    &= Z_{0}\left( 1 - \beta \langle \mathbf{H} \rangle_{0} +
        \frac{\beta^{2} \langle \mathbf{H}^{2}\rangle_{0}}{2} + \cdots \right)
```

$Z_{0}$ and $\langle \mathbf{H}^{n}\rangle_{0}$ are defined as

\begin{gather*}
    Z_{0} := \sum_{\{\mathbf{s}_{i}\}} 1 \\
    \langle \mathbf{H}^{n}\rangle_{0} := \frac{\sum_{\{\mathbf{s}_{i}\}} \mathbf{H}^{n}(\{\mathbf{r}_{i}\}, \{\mathbf{s}_{i}\})}{Z_{0}}
\end{gather*}

We substitute equation {eq}`eq_1st_expansion` for $\ln Z$ and expand $\ln (1 + x)$.
Then, we obtain the following equation.

\begin{align*}
    \ln Z &= \ln Z_{0} + \left( - \beta \langle \mathbf{H} \rangle_{0}
        + \frac{\beta^{2} \langle \mathbf{H}^{2} \rangle_{0}}{2} + \cdots \right)
        - \frac{1}{2} \left( - \beta \langle \mathbf{H} \rangle_{0}
        + \frac{\beta^{2} \langle \mathbf{H}^{2} \rangle_{0}}{2} + \cdots \right)^{2}
        + \cdots \\
    &= \ln Z_{0} - \beta \langle \mathbf{H} \rangle_{0}
        + \frac{\beta^{2}}{2} \left( \langle \mathbf{H}^{2}\rangle_{0}
        - \langle \mathbf{H} \rangle_{0}^{2} \right)
        + \mathcal{O}(\beta^{3}) \\
    &= \ln Z_{0} - \beta \langle \mathbf{H} \rangle_{0} + \mathcal{O}(\beta^{2})
\end{align*}

Therefore, the Helmholtz free energy of the paramagnetic system at high temperature is written as

\begin{align*}
    F &= - \beta^{-1} \ln Z \\
    &= - \beta^{-1} \ln Z_{0} + \langle \mathbf{H} \rangle_{0} + \mathcal{O}(\beta)
\end{align*}

## Avaraging

### Pair Feature

We denote the total energy of given atomic configuration $\{\mathbf{r}_{c}\}$ and
spin configuration $\{\mathbf{s}_{c}\}$ by $\mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\})$.

\begin{align*}
    \mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}) &= \sum_{n} \sum_{t} w_{n, t} \left(\sum_{i} d_{n, t}^{(i)} \right)\\
        &= \sum_{n} \sum_{t} w_{n, t} \left( \sum_{i} \sum_{j \in \mathcal{N}_{i, t}} f_{n}(r_{ij}) \right)
\end{align*}

Here, $\mathcal{N}_{i, t}$ is defined as

\begin{align*}
    \mathcal{N}_{i, t} := \left\{ j  = 1, 2, \cdots, N \mid j \neq i, r_{ij} \le r_{\mathrm{cutoff}}, \{s_{i}, s_{j}\} = t \right\}
\end{align*}

$N$ is the total number of atoms in the unitcell.

\begin{align*}
    \mathcal{D}_{2} := \left\{ \{i, j\} \mid 1 \le i, j \le N ,\, r_{ij} \le r_{\mathrm{cutoff}} \right\}
\end{align*}

Let $E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j\}}$ be the term in $\mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\})$ that only the atom $i$ and the atom $j$ involve in. Then,

\begin{gather*}
    \mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}) = \sum_{\{i, j\} \in \mathcal{D}_{2}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j\}} \\
    E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j\}} = \sum_{n} w_{n,\{s_{i},s_{j}\}} \left( f_{n}(r_{ij}) + f_{n}(r_{ji}) \right)
\end{gather*}

Therefore, $\langle \mathbf{H} \rangle_{0}$ is written as

\begin{align*}
    \langle \mathbf{H} \rangle_{0} &= \sum_{\{\mathbf{s}_{c}\}} \mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}) \\
    &= \sum_{\{i, j\} \in \mathcal{D}_{2}} \left( \sum_{\{\mathbf{s}_{c}\}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j\}} \right) \\
    &= \sum_{\{i, j\} \in \mathcal{D}_{2}} \sum_{n} \left( \sum_{\{\mathbf{s}_{c}\}} w_{n, \{s_{i}, s_{j}\}}(f_{n}(r_{ij})
        + f_{n}(r_{ji}))) \right) \\
    &= \sum_{\{i, j\} \in \mathcal{D}_{2}} \sum_{n} \tilde{w}_{n} \left[ f_{n}(r_{ij}) + f_{n}(r_{ji}) \right] \\
    &= \sum_{n} \tilde{w}_{n} \left[ \sum_{\{i, j\} \in \mathcal{D}_{2}} \left( f_{n}(r_{ij}) + f_{n}(r_{ji}) \right) \right] \\
    &= \sum_{n} \tilde{w}_{n} \left( \sum_{i} \sum_{j \in \mathcal{N}_{i, t}} f_{n}(r_{ij}) \right) \\
    &= \sum_{n} \tilde{w}_{n} \left( \sum_{i} d^{(i)}_{n} \right)
\end{align*}

$\tilde{w}_{n}$ can be calculated as follows

\begin{align*}
    \tilde{w}_{n} &= \frac{w_{n, \{\uparrow, \uparrow \}} + w_{n, \{\uparrow, \downarrow \}} + w_{n, \{\downarrow, \uparrow \}} + w_{n, \{\downarrow, \downarrow \}}}{4} \\
        &= \frac{w_{n, \{\uparrow, \uparrow \}} + 2 w_{n, \{\uparrow, \downarrow \}} + w_{n, \{\downarrow, \downarrow \}}}{4}
\end{align*}

### 2nd-order rotational invariant (Linear Term)

\begin{align*}
    \mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}) &= \sum_{n} \sum_{l=0}^{l^{(2)}_{\mathrm{max}}} \sum_{\mathbf{\tau} \in \mathcal{T}} w_{nl, \mathbf{\tau}}
        \left( \sum_{i} d_{nl,\, \mathbf{\tau}}^{(i)} \right) \\
    &= \sum_{n} \sum_{l=0}^{l^{(2)}_{\mathrm{max}}} \sum_{\mathbf{\tau}=(t_{1}, t_{2}) \in \mathcal{T}} w_{nl,\mathbf{\tau}} \left( \sum_{i} \sum_{m_{1}=-l}^{l} \sum_{m_{2}=-l}^{l} a_{nlm_{1},t_{1}}^{(i)} a_{nlm_{2},t_{2}}^{(i)} \right) \\
    &= \sum_{n} \sum_{l=0}^{l^{(2)}_{\mathrm{max}}} \sum_{\mathbf{\tau}=(t_{1}, t_{2}) \in \mathcal{T}} w_{nl,\, t_{1} t_{2}}
        \left[ \sum_{m_{1}=-l}^{l} \sum_{m_{2}=-l}^{l} C_{m_{1} m_{2}}^{l} \left( \sum_{i} \sum_{j \in \mathcal{N}_{i,t_{1}}} \sum_{k \in \mathcal{N}_{i,t_{2}}} f_{n}(r_{ij}) Y_{lm_{1}}(\hat{\mathbf{r}}_{ij}) f_{n}(r_{ik})Y_{lm_{2}}(\hat{\mathbf{r}}_{ik}) \right) \right]
\end{align*}

We define the following quantities.

\begin{align*}
    \mathcal{D}_{2} &:= \{ \{i, j\} \mid i, j = 1, 2, \cdots, N,\,\, 0 < r_{ij} < r_{\mathrm{cutoff}}\} \\
    \mathcal{D}_{3} &:= \{ \{i, j, k\} \mid i, j, k = 1, 2, \cdots, N,\,\,
        0 < r_{ij}, r_{ik} < r_{\mathrm{cutoff}}\}
\end{align*}

Let $E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \Delta}$ be the term in $\mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\})$ that only the atoms
in $\Delta \in \mathcal{D}_{n}(n = 2, 3)$ involve in. Then

\begin{align*}
    \mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}) &= \sum_{\{i, j\} \in \mathcal{D}_{2}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j\}}
        + \sum_{\{i, j, k\} \in \mathcal{D}_{3}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j, k\}} \\
\end{align*}

Here, each term is defined as follows.
\begin{align*}
    E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j\}} & := \sum_{n} \sum_{l=0}^{l^{(2)}_{\mathrm{max}}}
        \sum_{m_{1}=-l}^{l} \sum_{m_{2}=-l}^{l}
        \left[ G^{(i)}_{nlm_{1}m_{2}}(j, j)
        + G^{(j)}_{nlm_{1}m_{2}}(i, i) \right] \\
    E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j, k\}} & := \sum_{n} \sum_{l=0}^{l^{(2)}_{\mathrm{max}}}
        \sum_{m_{1}=-l}^{l} \sum_{m_{2}=-l}^{l} \left\{
        \left[ G^{(i)}_{nlm_{1}m_{2}}(j, k)
        + G^{(i)}_{nlm_{1}m_{2}}(k, j) \right] + \Theta (r_{\mathrm{cutoff}} - r_{jk})
        \left[ G^{(j)}_{nlm_{1}m_{2}}(i, k)
        + G^{(j)}_{nlm_{1}m_{2}}(k, i) + G^{(k)}_{nlm_{1}m_{2}}(i, j)
        + G^{(k)}_{nlm_{1}m_{2}}(j, i) \right] \right\}
\end{align*}

Note that,

\begin{align*}
    F^{(i)}_{nlm_{1}m_{2}}(x, y) & := C_{m_{1}m_{2}}^{l} f_{n}(r_{ix})Y_{lm_{1}}(\hat{\mathbf{r}}_{ix})
        f_{n}(r_{iy})Y_{lm_{2}}(\hat{\mathbf{r}}_{iy}) \\
    G^{(i)}_{nlm_{1}m_{2}}(x, y) & := w_{n(l,\{s_{i}, s_{x}\})
        (l,\{s_{i}, s_{y}\})} C_{m_{1}m_{2}}^{l} f_{n}(r_{ix})Y_{lm_{1}}(\hat{\mathbf{r}}_{ix})
        f_{n}(r_{iy})Y_{lm_{2}}(\hat{\mathbf{r}}_{iy}) \\
\end{align*}

Therefore, $\langle \mathbf{H} \rangle_{0}$ is written as

\begin{align*}
    \langle \mathbf{H} \rangle_{0} &= \sum_{\{\mathbf{s}_{c}\}} \mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}) \\
        &= \sum_{\{i,j\} \in \mathcal{D}_{2}} \left( \sum_{\{\mathbf{s}_{c}\}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\},\{i,j\}} \right)
            + \sum_{\{i,j,k\} \in \mathcal{D}_{3}} \left( \sum_{\{\mathbf{s}_{c}\}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\},\{i,j,k\}} \right) \\
        &= \sum_{\{i,j\} \in \mathcal{D}_{2}} \sum_{n} \sum^{l^{(2)}_{\mathrm{max}}}_{l=0}
            \sum_{m_{1}=-l}^{l} \sum_{m_{2}=-l}^{l} \tilde{w}^{(2)}_{nl}
            \left( F_{nlm_{1}m_{2}}^{(i)}(j, j) + F_{nlm_{1}m_{2}}^{(j)}(i, i) \right) \\
        & \hspace{6em} + \sum_{\{i,j, k\} \in \mathcal{D}_{3}} \sum_{n} \sum^{l^{(2)}_{\mathrm{max}}}_{l=0}
            \sum_{m_{1}=-l}^{l} \sum_{m_{2}=-l}^{l} \tilde{w}^{(3)}_{nl}
            \left\{ \left[ F_{nlm_{1}m_{2}}^{(i)}(j, k) + F_{nlm_{1}m_{2}}^{(i)}(j, k) \right] + \Theta(r_{\mathrm{cutoff}} - r_{jk}) \left[ F_{nlm_{1}m_{2}}^{(j)}(i, k) + F_{nlm_{1}m_{2}}^{(j)}(k, i) + F_{nlm_{1}m_{2}}^{(k)}(i, j) + F_{nlm_{1}m_{2}}^{(k)}(j, i) \right] \right\}
\end{align*}

$\tilde{w}^{(2)}_{nl},\, \tilde{w}^{(3)}_{nl}$ can be calculated as follows

\begin{align*}
    \tilde{w}^{(2)}_{nl} &= \frac{w_{n(l,\{\uparrow, \uparrow \})(l,\{\uparrow, \uparrow \})} + w_{n (l,\{\uparrow, \downarrow \})(l,\{\uparrow, \downarrow \})} + w_{n (l,\{\downarrow, \uparrow \})(l,\{\downarrow, \uparrow \})} + w_{n (l,\{\downarrow, \downarrow \})(l,\{\downarrow, \downarrow \})}}{4} \\
        &= \frac{w_{n (l,\{\uparrow, \uparrow \})(l,\{\uparrow, \uparrow \})} + 2 w_{n, (l,\{\uparrow, \downarrow \})(l,\{\uparrow, \downarrow \})} + w_{n (l,\{\downarrow, \downarrow \})(l,\{\downarrow, \downarrow \})}}{4} \\
    \tilde{w}^{(3)}_{nl} &= \frac{w_{n(l,\{\uparrow, \uparrow \})(l,\{\uparrow, \uparrow \})} + w_{n(l,\{\uparrow, \uparrow \})(l,\{\uparrow, \downarrow \})}+ w_{n (l,\{\uparrow, \downarrow \})(l,\{\uparrow, \downarrow \})} + w_{n (l,\{\uparrow, \downarrow \})(l,\{\uparrow, \uparrow \})} + w_{n (l,\{\downarrow, \uparrow \})(l,\{\downarrow, \uparrow \})} + w_{n (l,\{\downarrow, \uparrow \})(l,\{\downarrow, \downarrow \})} + w_{n (l,\{\downarrow, \downarrow \})(l,\{\downarrow, \downarrow \})} + w_{n (l,\{\downarrow, \downarrow \})(l,\{\downarrow, \uparrow \})} }{8} \\
        &= \frac{w_{n(l,\{\uparrow, \uparrow \})(l,\{\uparrow, \uparrow \})} + 2 w_{n(l,\{\uparrow, \uparrow \})(l,\{\uparrow, \downarrow \})} + 2 w_{n (l,\{\uparrow, \downarrow \})(l,\{\uparrow, \downarrow \})} + 2 w_{n (l,\{\downarrow, \uparrow \})(l,\{\downarrow, \downarrow \})} + w_{n (l,\{\downarrow, \downarrow \})(l,\{\downarrow, \downarrow \})}}{8} \\
\end{align*}

### 3rd-order rotational invariant (Linear Term)

$\mathcal{L}$ is defined as

\begin{align*}
    \mathcal{L} := \{ (l_{1}, l_{2}, l_{3}) \mid l_{i} = 0, 1, \cdots, l^{(i)}_{\mathrm{max}}(i = 1, 2, 3), \sum_{i=1}^{3} l_{i} = 0\,\, \mathrm{mod}\, 2 \}
\end{align*}

Then, $\mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\})$ is written as

\begin{align*}
    \mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}) &= \sum_{n} \sum_{\mathbf{k} \in \mathcal{L}}
        \sum_{\boldsymbol{\tau} \in \mathcal{T}_{\mathbf{k}}} w_{n\mathbf{k}, \boldsymbol{\tau}}
        \left( \sum_{i} d^{(i)}_{n\mathbf{k}, \boldsymbol{\tau}} \right) \\
    &= \sum_{n} \sum_{\mathbf{k} = (l_{1}, l_{2}, l_{3}) \in \mathcal{L}}
        \sum_{\boldsymbol{\tau} = (t_{1}, t_{2}, t_{3}) \in \mathcal{T}_{\mathbf{k}}}
        w_{n\mathbf{k}, \boldsymbol{\tau}} \left( \sum_{i} \sum_{m_{1}=-l_{1}}^{l_{1}}
        \sum_{m_{2}=-l_{2}}^{l_{2}} \sum_{m_{3}=-l_{3}}^{l_{3}} a^{(i)}_{nl_{1}m_{1}, t_{1}}
        a^{(i)}_{nl_{2}m_{2}, t_{2}} a^{(i)}_{nl_{3}m_{3}, t_{3}} \right) \\
    &= \sum_{n} \sum_{\mathbf{k} = (l_{1}, l_{2}, l_{3}) \in \mathcal{L}}
        \sum_{\boldsymbol{\tau} = (t_{1}, t_{2}, t_{3}) \in \mathcal{T}_{\mathbf{k}}}
        w_{n\mathbf{k}, \boldsymbol{\tau}} \left[ \sum_{m_{1}=-l_{1}}^{l_{1}}
        \sum_{m_{2}=-l_{2}}^{l_{2}} \sum_{m_{3}=-l_{3}}^{l_{3}} C_{m_{1} m_{2} m_{3}}^{l_{1} l_{2} l_{3}} \left( \sum_{i}
        \sum_{j \in \mathcal{N}_{i,t_{1}}} \sum_{k \in \mathcal{N}_{i,t_{2}}}
        \sum_{l \in \mathcal{N}_{i,t_{3}}}
        f_{n}(r_{ij}) Y_{l_{1}m_{1}}(\hat{\mathbf{r}}_{ij})
        f_{n}(r_{ik}) Y_{l_{2}m_{2}}(\hat{\mathbf{r}}_{ik})
        f_{n}(r_{il}) Y_{l_{3}m_{3}}(\hat{\mathbf{r}}_{il}) \right) \right]
\end{align*}

We define the following quantities.

\begin{align*}
    \mathcal{D}_{2} &:= \{ \{i, j\} \mid i, j = 1, 2, \cdots, N,\,\, 0 < r_{ij} < r_{\mathrm{cutoff}}\} \\
    \mathcal{D}_{3} &:= \{ \{i, j, k\} \mid i, j, k = 1, 2, \cdots, N,\,\,
        0 < r_{ij}, r_{ik} < r_{\mathrm{cutoff}}\} \\
    \mathcal{D}_{4} &:= \{ \{i, j, k, l\} \mid i, j, k, l = 1, 2, \cdots, N,\,\,
        0 < r_{ij}, r_{ik}, r_{il} < r_{\mathrm{cutoff}}\}
\end{align*}


Let $E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \Delta}$ be the term in $\mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\})$ that only the atoms
in $\Delta \in \mathcal{D}_{n}(n = 2, 3, 4)$ involve in. Then

\begin{align*}
    \mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}) &= \sum_{\{i, j\} \in \mathcal{D}_{2}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j\}}
        + \sum_{\{i, j, k\} \in \mathcal{D}_{3}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j, k\}}
        + \sum_{\{i, j, k, l\} \in \mathcal{D}_{4}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j, k, l\}} \\
\end{align*}

Here, each term is defined as follows.
\begin{align*}
    E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j\}} & := \sum_{n} \sum_{\mathbf{k}=(l_{1}, l_{2}, l_{3}) \in \mathcal{L}}
        \sum_{m_{1}=-l_{1}}^{l_{1}} \sum_{m_{2}=-l_{2}}^{l_{2}}
        \sum_{m_{3}=-l_{3}}^{l_{3}}
        \left[ G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, j, j)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, i, i) \right] \\
    E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j, k\}} & := \sum_{n} \sum_{\mathbf{k}=(l_{1}, l_{2}, l_{3}) \in \mathcal{L}}
        \sum_{m_{1}=-l_{1}}^{l_{1}} \sum_{m_{2}=-l_{2}}^{l_{2}}
        \sum_{m_{3}=-l_{3}}^{l_{3}} \{
        [ G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, k, k)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, j, k)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, k, j)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, j, j)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, k, j)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, j, k) ] \\
        & \hspace{22em} + \Theta (r_{\mathrm{cutoff}} - r_{jk})
        [ G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, k, k)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, i, k)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, k, i)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, i, i)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, k, i)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, i, k) \\
        & \hspace{22em} + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, j, j)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, i, j)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, j, i)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, i, i)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, j, i)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, i, j) ]\} \\
    E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i, j, k, l\}} & := \sum_{n}
        \sum_{\mathbf{k}=(l_{1}, l_{2}, l_{3}) \in \mathcal{L}} \sum_{m_{1}=-l_{1}}^{l_{1}}
        \sum_{m_{2}=-l_{2}}^{l_{2}} \sum_{m_{3}=-l_{3}}^{l_{3}}
        \{[ G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, k, l)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, l, k)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, j, l)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, l, j)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(l, j, k)
        + G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(l, k, j)] \\
        & \hspace{22em} +
        \Theta (r_{\mathrm{cutoff}} - r_{lj}) \Theta (r_{\mathrm{cutoff}} - r_{jk})
        [ G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, k, l)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, l, k)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, i, l)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, l, i)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(l, i, k)
        + G^{(j)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(l, k, i)] \\
        & \hspace{22em} + \Theta (r_{\mathrm{cutoff}} - r_{jk}) \Theta (r_{\mathrm{cutoff}} - r_{kl})
        [ G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, j, l)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, l, j)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, i, l)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, l, i)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(l, i, j)
        + G^{(k)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(l, j, i)] \\
        & \hspace{22em} + \Theta (r_{\mathrm{cutoff}} - r_{kl}) \Theta (r_{\mathrm{cutoff}} - r_{lj})
        [ G^{(l)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, j, k)
        + G^{(l)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(i, k, j)
        + G^{(l)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, i, k)
        + G^{(l)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(j, k, i)
        + G^{(l)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, i, j)
        + G^{(l)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(k, j, i)]\} \\
\end{align*}

Note that,

\begin{align*}
    F^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(x, y, z) & := C_{m_{1}m_{2}m_{3}}^{l_{1}l_{2}l_{3}} f_{n}(r_{ix})Y_{l_{1}m_{1}}(\hat{\mathbf{r}}_{ix})
        f_{n}(r_{iy})Y_{l_{2}m_{2}}(\hat{\mathbf{r}}_{iy}) f_{n}(r_{iz})Y_{l_{3}m_{3}}(\hat{\mathbf{r}}_{iz}) \\
    G^{(i)}_{nl_{1}l_{2}l_{3}m_{1}m_{2}m_{3}}(x, y, z) & := C_{m_{1}m_{2}m_{3}}^{l_{1}l_{2}l_{3}} w_{n(l_{1},\{s_{i}, s_{x}\})
        (l_{2},\{s_{i}, s_{y}\})(l_{3}, \{s_{i}, s_{z}\})} f_{n}(r_{ix})Y_{l_{1}m_{1}}(\hat{\mathbf{r}}_{ix})
        f_{n}(r_{iy})Y_{l_{2}m_{2}}(\hat{\mathbf{r}}_{iy}) f_{n}(r_{iz})Y_{l_{3}m_{3}}(\hat{\mathbf{r}}_{iz}) \\
\end{align*}

Therefore, $\langle \mathbf{H} \rangle_{0}$ is written as

\begin{align*}
    \langle \mathbf{H} \rangle_{0} &= \sum_{\{\mathbf{s}_{c}\}} \mathbf{H}(\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}) \\
        &= \sum_{\{i,j\} \in \mathcal{D}_{2}} \left( \sum_{\{\mathbf{s}_{c}\}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\} ,\{i,j\}} \right)
            + \sum_{\{i,j,k\} \in \mathcal{D}_{3}} \left( \sum_{\{\mathbf{s}_{c}\}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i,j,k\}} \right)
            + \sum_{\{i,j,k,l\} \in \mathcal{D}_{4}} \left( \sum_{\{\mathbf{s}_{c}\}} E_{\{\mathbf{r}_{c}\}, \{\mathbf{s}_{c}\}, \{i,j,k,l\}} \right)
\end{align*}


## Loss Function

We defined loss function.

\begin{equation*}
    \mathcal{L} := (1 - \alpha) \times \mathcal{L}_{E} + \alpha \times \mathcal{L}_{F}
\end{equation*}

Here, $\mathcal{L}_{E}$ and $\mathcal{L}_{F}$ are defined as follows.

\begin{align*}
    \mathcal{L}_{E} &:= \sqrt{\frac{1}{N}\sum_{i}\frac{(\hat{E}^{(i)} - E^{(i)})^{2}}{\sigma(E)^{2}}} \\
                    &= \frac{\mathrm{RMSE}(E)}{\sigma(E)}
\end{align*}

\begin{align*}
    \mathcal{L}_{F} &:= \sqrt{\frac{1}{N}\sum_{i}\frac{1}{N_{\mathrm{atom}}^{(i)}}\sum_{j}\frac{|\hat{\mathbf{F}}_{j}^{(i)} - \mathbf{F}_{j}^{(i)}|^{2}}{\sigma(F)^{2}}} \\
                    &= \frac{\sqrt{3}\,\mathrm{RMSE}(F)}{\sigma(F)}
\end{align*}
