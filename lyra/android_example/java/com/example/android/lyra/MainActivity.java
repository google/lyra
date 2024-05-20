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
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * The main activity for the Lyra android example.
 * It features benchmarking to logcat and coding lyra from the mic.
 */
public class MainActivity extends AppCompatActivity {
  private static final String TAG = "MainActivity";

  static {
    System.loadLibrary("lyra_android_example");
  }

  private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
  private static final int SAMPLE_RATE = 16000;
  private static final int LYRA_NUM_RANDOM_FEATURE_VECTORS = 10000;
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

    // Populate the bits per second dropdown widget.
    Spinner spinner = (Spinner) findViewById(R.id.bps_spinner);
    Integer[] bpsArray = new Integer[]{3200, 6000, 9200};
    ArrayAdapter<Integer> adapter =
        new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, bpsArray);
    adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
    spinner.setAdapter(adapter);

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

  private synchronized void encodeAndDecodeMicDataToSpeaker(int bitrate) {
    // There must be at least enough data recorded to output something useful.
    if (micDataShortsWritten == 0) {
      return;
    }
    // Encode the recorded wave inside the micData with Lyra
    short[] encodedAudio = encodeSamples(micData, micDataShortsWritten, SAMPLE_RATE, bitrate,
        weightsDirectory);

    // Notes for Chase: 
    // The bytes in "encodedAudio" array contains all the data you wish to transmit via our LoRa network.
    // The length of the short array "encodedAudio" is typically larger than the maximum payload
    // size allowed by our LoRa protocol, so make sure you manually separate this array into several
    // small packets (e.g. ~100 bytes long) and transmit them via LoRa network sequentially. 
    // 
    // Another thing to note is that the Lyra codec can tolarate packet losses up to some level. Therefore, it 
    // might be useful to try different sizes of the small packets to find which size(s) give you the best audio
    // experience when packet loss is present.
    // 
    // At the receiving side, when you receive a packet, you can call "decodeSample" function to decode a Lyra packet.
    // The decoded audio should be buffered into an array first and then play the buffered audio (see the example below).
    // In practice, do not assume that the receiving party will receive 100% of the transmitted packets. Therefore, just
    // decode what you receive - the interrupting audio may still be useful to the end users.
    short[] decodedAudio = decodeSamples(encodedAudio, micDataShortsWritten, SAMPLE_RATE,
                                         bitrate, weightsDirectory);
    if (decodedAudio == null) {
      Log.e(TAG, "Failed to encode and decode microphone data.");
      return;
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
            0,
            decodedAudio.length,
            AudioTrack.WRITE_BLOCKING);
    Log.i(
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
    button.post(() -> button.setText(R.string.button_record));
    Button decodeButton = (Button) findViewById(R.id.button_decode);
    decodeButton.setEnabled(true);
  }

  /** Called when user taps the 'Encode/Decode To Speaker' button. */
  public void onDecodeButtonClicked(View view) {
    Log.i(TAG, "Starting decoding.");

    Button decodeButton = (Button) view;
    decodeButton.setEnabled(false);
    Button recordButton = (Button) findViewById(R.id.button_record);
    recordButton.setEnabled(false);

    Spinner bpsSpinner = (Spinner) findViewById(R.id.bps_spinner);
    int bps = Integer.parseInt(bpsSpinner.getSelectedItem().toString());
    MainActivity mainActivity = this;
    Thread thread =
        new Thread(
            () -> {
              encodeAndDecodeMicDataToSpeaker(bps);
              mainActivity.runOnUiThread(
                  () -> {
                    decodeButton.setEnabled(true);
                    recordButton.setEnabled(true);
                  });
            });
    thread.start();
  }

  /** Called when user taps the 'record microphone' button. */
  public void onMicButtonClicked(View view) {
    if (!isRecording) {
      isRecording = true;
      // Begin recording, and set the button to be a stop button.
      ((Button) view).setText(R.string.button_stop);
      Button decodeButton = (Button) findViewById(R.id.button_decode);
      decodeButton.setEnabled(false);
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
    }
  }

  /** Called when user taps the benchmark button. */
  public void runBenchmark(View view) {
    if (!hasStartedDecode) {
      TextView tv = (TextView) findViewById(R.id.sample_text);
      Button button = (Button) view;
      button.setEnabled(false);
      tv.setText(R.string.benchmark_in_progress);
      hasStartedDecode = true;

      new Thread(
              () -> {
                Log.i(TAG, "Starting lyraBenchmark()");
                // Example of a call to a C++ lyra method on a background
                // thread.
                lyraBenchmark(LYRA_NUM_RANDOM_FEATURE_VECTORS, weightsDirectory);
                Log.i(TAG, "Finished lyraBenchmark()");
                tv.post(() -> tv.setText(R.string.benchmark_finished));
                button.post(() -> button.setEnabled(true));
                hasStartedDecode = false;
              })
          .start();
    }
  }

  private void copyWeightsAssetsToDirectory(String targetDirectory) {
    try {
      AssetManager assetManager = getAssets();
      String[] files = {"lyra_config.binarypb", "lyragan.tflite",
        "quantizer.tflite", "soundstream_encoder.tflite"};
      byte[] buffer = new byte[1024];
      int amountRead;
      for (String file : files) {
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
   * Methods that are implemented by the 'lyra_android_example' C++ library, which is packaged with
   * this application.
   */
  public native String lyraBenchmark(int numCondVectors, String modelBasePath);

  public native short[] encodeSamples(
    short[] samples, int sampleLength, int sample_rate_Hz, 
    int bitrate, String modelBasePath);

  public native short[] decodeSamples(
    short[] samples, int sampleLength, int sample_rate_Hz, 
    int bitrate, String modelBasePath);
}
