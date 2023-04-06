---
layout: page
title: "Experiment 1"
permalink: /experiment-1

---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

#### Description

Effect of training window length, $$W$$, on model accuracy on three case studies:

a) Digital phaser, $$T_0=2\text{s}$$

b) EHX Small Stone, rate = 3o'clock, Colour Off (SS-A)

c) EHX Small Stone, rate = 3o'clock, Colour On (SS-A)

#### Guitar samples (testing data)
<table>
  <thead>
    <tr>
      <th>System</th>
      <th>Reference</th>
      <th>W=10ms</th>
      <th>W=20ms</th>
      <th>W=40ms</th>
      <th>W=80ms</th>
      <th>W=160ms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DP-2</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_full_phaser_rate=0p5_fb=0p7_saw.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_full_W=10ms_3f70zm7c.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_full_W=20ms_29ayt2gj.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_full_W=40ms_2b1qi9re.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_full_W=80ms_1h2lwyr8.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_full_W=160ms_li4p2nw1.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
      <td>SS-A</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_full_colour=0_rate=3oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_full_W=10ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_full_W=20ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_full_W=40ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_full_W=80ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_full_W=160ms.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
      <td>SS-D</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_full_colour=1_rate=3oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_full_W=10ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_full_W=20ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_full_W=40ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_full_W=80ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_full_W=160ms.mp3" type="audio/mp3">
        </audio></td>
    </tr>
  </tbody>
</table>

#### Chirp-train samples (training data)
<table>
  <thead>
    <tr>
      <th>System</th>
      <th>Reference</th>
      <th>W=10ms</th>
      <th>W=20ms</th>
      <th>W=40ms</th>
      <th>W=80ms</th>
      <th>W=160ms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DP-2</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_train_phaser_rate=0p5_fb=0p7_saw.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_train_W=10ms_3f70zm7c.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_train_W=20ms_29ayt2gj.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_train_W=40ms_2b1qi9re.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_train_W=80ms_1h2lwyr8.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/DP-2_train_W=160ms_li4p2nw1.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
      <td>SS-A</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_train_colour=0_rate=3oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_train_W=10ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_train_W=20ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_train_W=40ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_train_W=80ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-A_train_W=160ms.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
      <td>SS-D</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_train_colour=1_rate=3oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_train_W=10ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_train_W=20ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_train_W=40ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_train_W=80ms.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/SS-D_train_W=160ms.mp3" type="audio/mp3">
        </audio></td>
    </tr>
  </tbody>
</table>


#### Dry audio samples
<table>
  <thead>
    <tr>
      <th>Audio Clip</th>
      <th>Dry input</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Chirp train (training data)</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp1/train_input_dry.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
      <td>Guitar (testing data)</td>
      <td>
        <audio controls>
          <source src="audio-examples/dry/full_input_dry.mp3" type="audio/mp3">
        </audio></td>
    </tr>
  </tbody>
</table>



