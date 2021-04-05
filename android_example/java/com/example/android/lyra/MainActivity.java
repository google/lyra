// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.example.android.lyra;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.support.v4.app.ActivityCompat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
  private static final String TAG = "MainActivity";

  static {
    System.loadLibrary("lyra_android_example");
  }

  private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
  private static final int SAMPLE_RATE = 16000;
  private static final int PLAYBACK_SKIP_SAMPLES = 4000;
  private static final String[] permissions = {Manifest.permission.RECORD_AUDIO};

  private boolean hasStartedDecode = false;
  private boolean isRecording = false;
  private String weightsDirectory;
  private AudioRecord record;
  private AudioTrack player;
  private short[] micData;
  private int micDataShortsWritten;

  // Requesting permission to RECORD_AUDIO
  private boolean permissionToRecordAccepted = false;

  @Override
  public void onRequestPermissionsResult(
      int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    switch (requestCode) {
      case REQUEST_RECORD_AUDIO_PERMISSION:
        permissionToRecordAccepted = grantResults[0] == PackageManager.PERMISSION_GRANTED;
        break;
      default:
        throw new AssertionError("Unhandled permission code: " + requestCode);
    }
    if (!permissionToRecordAccepted) {
      finish();
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    // The weights are stored inside of the APK as assets for this demo, but
    // the Lyra library requires them to live in files.
    // This helper function copies the assets to files.
    // It is not necessarily the case that you should have the weights as assets.
    // For example, your application might download the weights from a server
    // instead, in which case they would only exist as files.
    weightsDirectory = getExternalFilesDir(null).getAbsolutePath();
    copyWeightsAssetsToDirectory(weightsDirectory);

    // This demo uses the microphone, which we need permission for.
    ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);
  }

  private synchronized void recordAudioStream() {
    Log.i(TAG, "Starting recording from microphone.");

    // This example records and encodes in series, to minimize complexity.
    final int chunkSize = 1000;
    if (micData == null) {
      micData = new short[SAMPLE_RATE * 5 + chunkSize];
    }
    micDataShortsWritten = 0;
    while (isRecording) {
      // If we are not yet full, write the wav data;
      if (micDataShortsWritten <= micData.length - chunkSize) {
        int amountRead =
            record.read(micData, micDataShortsWritten, chunkSize, AudioRecord.READ_NON_BLOCKING);
        micDataShortsWritten += amountRead;
      }
    }

    // Recording has stopped.  Encoding/decoding will happen later.
    record.release();
    record = null;
    Log.i(
        TAG, "Finished recording from microphone.  Recorded " + micDataShortsWritten + " samples.");
  }

  private synchronized void encodeAndDecodeMicDataToSpeaker() {
    // There must be at least enough data recorded to output something useful.
    if (micDataShortsWritten < PLAYBACK_SKIP_SAMPLES) {
      return;
    }
    // Whatever micData holds, encode and decode with Lyra.
    short[] decodedAudio = encodeAndDecodeSamples(micData, micDataShortsWritten, weightsDirectory);

    if (decodedAudio == null) {
      Log.e(TAG, "Failed to encode and decode microphone data.");
    }

    // Create a new AudioTrack in static mode so we can write once and
    // replay it.
    AudioTrack player =
        new AudioTrack.Builder()
            .setAudioAttributes(
                new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build())
            .setTransferMode(AudioTrack.MODE_STATIC)
            .setAudioFormat(
                new AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build())
            .setBufferSizeInBytes(micData.length * 2)
            .build();

    // Skip the first quarter second because it contains transient noise.
    int shortsWritten =
        player.write(
            decodedAudio,
            PLAYBACK_SKIP_SAMPLES,
            decodedAudio.length - PLAYBACK_SKIP_SAMPLES,
            AudioTrack.WRITE_BLOCKING);
    Log.e(
        TAG,
        "Wrote "
            + shortsWritten
            + " of total length "
            + decodedAudio.length
            + " samples to AudioTrack.");
    player.play();
  }

  private void stopRecording() {
    record.stop();
    isRecording = false;
    // Notify we stopped recording.
    Button button = (Button) findViewById(R.id.button_record);
    button.post(() -> button.setText("Record from microphone"));
  }

  /** Called when user taps the 'record microphone' button. */
  public void onMicButtonClicked(View view) {
    if (!isRecording) {
      isRecording = true;
      // Begin recording, and set the button to be a stop button.
      ((Button) view).setText("Stop and decode to speaker");
      record =
          new AudioRecord.Builder()
              .setAudioSource(MediaRecorder.AudioSource.VOICE_COMMUNICATION)
              .setAudioFormat(
                  new AudioFormat.Builder()
                      .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                      .setSampleRate(SAMPLE_RATE)
                      .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                      .build())
              .setBufferSizeInBytes(1024 * 256)
              .build();
      record.startRecording();
      new Thread(this::recordAudioStream).start();
    } else {
      stopRecording();
      encodeAndDecodeMicDataToSpeaker();
    }
  }

  /** Called when user taps the benchmark button. */
  public void runBenchmark(View view) {
    if (!hasStartedDecode) {
      TextView tv = (TextView) findViewById(R.id.sample_text);
      Button button = (Button) view;
      button.setEnabled(false);
      tv.setText("Benchmark in progress. See logcat output for progress.");
      hasStartedDecode = true;

      new Thread(
              () -> {
                Log.i(TAG, "Starting benchmarkDecode()");
                // Example of a call to a C++ lyra method on a background
                // thread.
                benchmarkDecode(2000, weightsDirectory);
                Log.i(TAG, "Finished benchmarkDecode()");
                tv.post(() -> tv.setText("Finished benchmarking. See logcat for results."));
                button.post(() -> button.setEnabled(true));
                hasStartedDecode = false;
              })
          .start();
    }
  }

  private void copyWeightsAssetsToDirectory(String targetDirectory) {
    try {
      AssetManager assetManager = getAssets();
      String[] files = assetManager.list("");
      byte[] buffer = new byte[1024];
      int amountRead;
      for (String file : files) {
        // Lyra weights start with a 'lyra_' prefix.
        if (!file.startsWith("lyra_")) {
          continue;
        }
        InputStream inputStream = assetManager.open(file);
        File outputFile = new File(targetDirectory, file);

        OutputStream outputStream = new FileOutputStream(outputFile);
        Log.i(TAG, "copying asset to " + outputFile.getPath());

        while ((amountRead = inputStream.read(buffer)) != -1) {
          outputStream.write(buffer, 0, amountRead);
        }
        inputStream.close();
        outputStream.close();
      }
    } catch (Exception e) {
      Log.e(TAG, "Error copying assets", e);
    }
  }

  /**
   * A method that is implemented by the 'lyra_android_example' C++ library, which is packaged with
   * this application.
   */
  public native String benchmarkDecode(int numCondVectors, String modelBasePath);

  public native short[] encodeAndDecodeSamples(
      short[] samples, int sampleLength, String modelBasePath);
}
