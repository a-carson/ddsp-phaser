---
layout: page
title: "Experiment 2"
permalink: /experiment-2

---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

#### Description

This experiment considered training the models against the digital phaser data for different LFO rates:
fast ($$T_0=0.5\text{s}$$), medium ($$T_0=2\text{s}$$) and slow ($$T_0=8\text{s}$$).

The training window lengths, $$W$$, were set relative to the target LFO period:

$$ W = T_0 2^b / 100 $$

The audio examples below demonstrate  the shortest, median and longest window lengths 
used for each rate.


#### Digital Phaser
Target parameters: feedforward gain: $$g_1 = 1.0$$,
feedback gain: $$g_2=0.7$$,
delay-line length $$\phi=1$$.

<table>
  <thead>
    <tr>
      <th>$$T_0$$</th>
      <th>Reference</th>
      <th>$$b=0, \quad W/T_0 = 1\%$$</th>
      <th>$$b=2.5, \quad W/T_0 \approx 5\%$$</th>
      <th>$$b=5, \quad W/T_0 = 32\%$$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.5s</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-0.5_full_target.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-0.5_full_W=5ms_2hjwiefa.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-0.5_full_W=28ms_f1aggnqp.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-0.5_full_W=160ms_3mk15rlq.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
    <td>2s</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-2_full_target.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-2_full_W=20ms_29ayt2gj.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-2_full_W=113ms_3e8u3thn.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-2_full_W=640msdecv34yl.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
      <td>8s</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-8_full_target.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-8_full_W=113ms_1flfwy5e.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-8_full_W=453ms_a0ym85ke.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp2/DP-8_full_W=2.56s_241p7yra.mp3" type="audio/mp3">
        </audio></td>
    </tr>

  </tbody>
</table>

