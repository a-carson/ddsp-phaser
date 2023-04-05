---
layout: page
title: "Experiment 3"
permalink: /experiment-3

---

#### Description

The audio examples below demonstrate the effect of inference window length, W, on model accuracy when emulating the Electro-Harmonix (EHX) Colour Stone phaser pedal.

#### EHX Small-stone -- Colour Off (no feedback)

<table>
  <thead>
    <tr>
      <th>Rate</th>
      <th>Reference</th>
      <th>W=5ms</th>
      <th>W=40ms</th>
      <th>W=320ms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Fast ss-A</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-A_full_colour=0_rate=3oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-A_full_3dkbquzk_0.005.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-A_full_3dkbquzk_0.040.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-A_full_3dkbquzk_0.320.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
    <td>Med ss-B</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-B_full_colour=0_rate=12oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-B_full_ina3t8sx_0.005.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-B_full_ina3t8sx_0.040.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-B_full_ina3t8sx_0.320.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
      <td>Slow ss-C</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-C_full_colour=0_rate=9oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-C_full_3swsng10_0.005.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-C_full_3swsng10_0.040.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-C_full_3swsng10_0.320.mp3" type="audio/mp3">
        </audio></td>
    </tr>

  </tbody>
</table>


#### EHX Small-stone -- Colour On (with feedback)

<table>
  <thead>
    <tr>
      <th>Rate</th>
      <th>Reference</th>
      <th>W=5ms</th>
      <th>W=40ms</th>
      <th>W=320ms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> Fast ss-D </td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-D_full_colour=1_rate=3oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-D_full_3tlnx84b_0.005.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-D_full_3tlnx84b_0.040.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-D_full_3tlnx84b_0.320.mp3" type="audio/mp3">
        </audio></td>
    </tr>
    <tr>
      <td>Med ss-E</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-E_full_colour=1_rate=12oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-E_full_20v0y9md_0.005.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-E_full_20v0y9md_0.040.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-E_full_20v0y9md_0.320.mp3" type="audio/mp3">
        </audio></td>
    </tr>
   <tr>
      <td>Slow ss-F</td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-F_full_colour=1_rate=9oclock.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-F_full_34u8dt0q_0.005.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-F_full_34u8dt0q_0.040.mp3" type="audio/mp3">
        </audio></td>
      <td>
        <audio controls>
          <source src="audio-examples/exp3/SS-F_full_34u8dt0q_0.320.mp3" type="audio/mp3">
        </audio></td>
    </tr>
  </tbody>
</table>
