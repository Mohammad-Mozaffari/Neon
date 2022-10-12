# Conjugate Gradient Algorithm
The pseudocode of the main body of the conjugate gradient (CG) algorithm is as follows:

    1- beta := delta_new/delta_old (computed on the fly inside updateP container)

    2- p := r + beta*p (updateP container): Map

    3- s := Ap (matVec container): Stencil

    4- pAp := <p,s> (dot container): Reduction

    5- alpha := delta_new/pAp (computed on the fly inside updateXandR container)

    6- x := x + alpha*p (updateXandR container): Map

    7- r := r - alpha*S (updateXandR container): Map

    8- delta_old := delta_new (done inside updateXandR container)

    9- delta_new := <r,r> (dot container)

We aim to fuse different parts of this algorithm.

## Map-Stencil Fusion:
Steps 2 (map) and 3 (stencil) steps in the CG algorithm can be fused. The number of memory reads, writes, halo updates, and flops of each step is discussed below. In the analysis we assume that we have and $(n, n)$ grid, and we have $g$ GPUs.

### Map:
Step 2 works on two fields, `p` and `r`. 

\#Reads: $2  n^2$

\#Writes: $n^2$

\#Flops: $n^2$
* Only considering multiplications as flops. Note that there's no fused multiplication and add in this step.

### Stencil:
Step 3 works on the `p` field.

\#Reads: No Cache Reuse: $5 n^2$ - Full Cache Reuse: $n^2$
* Note that this number can be added by $n^2$ if the values of filed `s` will be loaded before the store.

\#Writes: $n^2$

\#Flopa: $5n^2$

\#Halo Update Elements per GPU: $2n$
* Note that the boundary GPUs will have $n$ elements.

### Baseline:
Combining the separate map and stencil costs we'll have:
\#Reads: No Cache Reuse: $7 n^2$ - Full Cache Reuse: $3 n^2$

\#Writes: 

### Fused Map-Stencil:
The psuedocode for the fused version is as follows:

    2.1- p2 := r + beta * p1
    2.2- s := A(r + beta * p1)
    2.3- swap(p1, p2)

*Please note that we have to use two fields `p1` and `p2` for storing `p` as `p` has a loop in its compute graph.

\#Reads: No Cache Reuse: $10 n^2$ - Full Cache Reuse: $2 n^2$

\#Writes: $2n^2$

\#Flops: $10 n^2$
* Each element in 2.2 requires 5 fused multiplication and adds, where each of them requires another multiplication.
* The computation of 2.1 is reused in 2.2.

\#Halo Update Elements per GPU: $4 n$
* The boundary GPUs will have $2n$ elements.


Asuming  $m = n^2 $, the following table compares the baseline and the fused version.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">#Reads - No Cache Reuse</th>
    <th class="tg-0pky">#Reads - Full Cache Reuse</th>
    <th class="tg-0lax">#Writes</th>
    <th class="tg-0lax">#Flops</th>
    <th class="tg-0lax">#Halo Update Points / GPU</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">Baseline</td>
    <td class="tg-0lax">7m or 8m</td>
    <td class="tg-0lax">3m or 4m</td>
    <td class="tg-0lax">2m</td>
    <td class="tg-0lax">6m</td>
    <td class="tg-0lax">2n</td>
  </tr>
  <tr>
    <td class="tg-0lax">Fused</td>
    <td class="tg-0lax">10m or 12m</td>
    <td class="tg-0lax">2m or 4m</td>
    <td class="tg-0lax">2m</td>
    <td class="tg-0lax">10m</td>
    <td class="tg-0lax">4n</td>
  </tr>
</tbody>
</table>


<!-- ## Reduciton Optimizaiton
We can use the value of the field to be reduced while `p` and `s` are loaded and computed in the fast memory.  Using this optimization, we'll require an extra field to store the field that will be reduced, but -->