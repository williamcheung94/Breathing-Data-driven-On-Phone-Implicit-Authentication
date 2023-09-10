/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Demonstrates how to run an audio recognition model in Android.

This example loads a simple speech recognition model trained by the tutorial at
https://www.tensorflow.org/tutorials/audio_training

The model files should be downloaded automatically from the TensorFlow website,
but if you have a custom model you can update the LABEL_FILENAME and
MODEL_FILENAME constants to point to your own files.

The example application displays a list view with all of the known audio labels,
and highlights each one when it thinks it has detected one through the
microphone. The averaging of results to give a more reliable signal happens in
the RecognizeCommands helper class.
*/

package org.tensorflow.lite.examples.speech;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.NoiseSuppressor;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import androidx.annotation.NonNull;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.content.ContextCompat;

import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

//import com.chaquo.python.PyObject;
//import com.chaquo.python.Python;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import java.util.Random;
import org.tensorflow.lite.Interpreter;

import io.socket.client.IO;
import io.socket.client.Socket;

public class SpeechActivity extends Activity
        implements View.OnClickListener, CompoundButton.OnCheckedChangeListener {

  private static final String POS_COUNTER_HEADER = "Accept User count: ";
  private static final String NEG_COUNTER_HEADER = "Reject User count: ";
  private static final String POS_COUNTER = "Accept User: ";
  private static final String NEG_COUNTER = "Reject User: ";
  private int pos_count = 0;
  private int neg_count = 0;
  private int pos = 0;
  private int neg = 0;
  Button pos_counter;
  Button neg_counter;
  Button foreground_button;
  TextView txv_pos;
  TextView txv_neg;
  ArrayList<String> micInput = new ArrayList<String>();
  ArrayList<String> userInput = new ArrayList<String>();
  ArrayList<Long> timeInput = new ArrayList<Long>();
  ArrayList<Long> startInput = new ArrayList<Long>();
  long ut = 0;
  long ut_start = 0;

  private static final int SAMPLE_RATE = 16000;
  private static final int SAMPLE_DURATION_MS = 1000;
  private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
  private static final long AVERAGE_WINDOW_DURATION_MS = 1000; //500
  private static final float DETECTION_THRESHOLD = 0.35f;//0.50f
  private static final int SUPPRESSION_MS = 1500;
  private static final int MINIMUM_COUNT = 3;
  private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30; //30
  private static final String LABEL_FILENAME = "file:///android_asset/conv_actions_labels.txt";
  private static final String MODEL_FILENAME = "file:///android_asset/conv_actions_frozen.tflite";

  private static final int REQUEST_RECORD_AUDIO = 13;
  private static final String LOG_TAG = SpeechActivity.class.getSimpleName();

  short[] recordingBuffer = new short[RECORDING_LENGTH];
  short[] copyForSaving = new short[RECORDING_LENGTH];
  int recordingOffset = 0;
  boolean shouldContinue = true;
  private Thread recordingThread;
  boolean shouldContinueRecognition = true;
  private Thread recognitionThread;
  private final ReentrantLock recordingBufferLock = new ReentrantLock();

  private List<String> labels = new ArrayList<String>();
  private List<String> displayedLabels = new ArrayList<>();
  private RecognizeCommands recognizeCommands = null;
  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  private Interpreter tfLite;
  private ImageView bottomSheetArrowImageView;

  private TextView yesTextView,
          noTextView;

  private TextView sampleRateTextView, inferenceTimeTextView;
  private ImageView plusImageView, minusImageView;
  private SwitchCompat apiSwitchCompat;
  private TextView threadsTextView;
  private long lastProcessingTimeMs;
  private Handler handler = new Handler();
  private TextView selectedTextView = null;
  private HandlerThread backgroundThread;
  private Handler backgroundHandler;

  private static final String FILE_NAME_User = "User_Results.txt";
  private static final String FILE_NAME_Input = "Input_Results.txt";
  private String mic_input = "-1";
  private  String user_input = "-1";
  private static final String CHAT_SERVER_URL = "http://192.168.1.104:8000/";
  Socket mSocket = null;

  long utName = 0;
  byte[] Data = new byte[SAMPLE_RATE];

  private Boolean isSpeech = false;
  final SpeechRecognizer mSpeechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
  final Intent mSpeechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);


  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
          throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  private void rawToWave(final File rawFile, final File waveFile)
    throws IOException{
    byte[] rawData = new byte[(int) rawFile.length()];
    DataInputStream input = null;
    try {
      input = new DataInputStream(new FileInputStream(rawFile));
      input.read(rawData);
    }finally {
      if (input != null) {
        input.close();
      }
    }
    DataOutputStream output = null;
    try {
      output = new DataOutputStream(new FileOutputStream(waveFile));
      // WAVE header
      // see http://ccrma.stanford.edu/courses/422/projects/WaveFormat/
      writeString(output, "RIFF"); // chunk id
      writeInt(output, 36 + rawData.length); // chunk size
      writeString(output, "WAVE"); // format
      writeString(output, "fmt "); // subchunk 1 id
      writeInt(output, 16); // subchunk 1 size
      writeShort(output, (short) 1); // audio format (1 = PCM)
      writeShort(output, (short) 1); // number of channels
      writeInt(output, 44100); // sample rate
      writeInt(output, SAMPLE_RATE * 2); // byte rate
      writeShort(output, (short) 2); // block align
      writeShort(output, (short) 16); // bits per sample
      writeString(output, "data"); // subchunk 2 id
      writeInt(output, rawData.length); // subchunk 2 size
      // Audio data (conversion big endian -> little endian)
      short[] shorts = new short[rawData.length / 2];
      ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);
      ByteBuffer bytes = ByteBuffer.allocate(shorts.length * 2);
      for (short s : shorts) {
        bytes.putShort(s);
      }

      output.write(fullyReadFileToBytes(rawFile));
    } finally {
      if (output != null) {
        output.close();
      }
    }
  }
  byte[] fullyReadFileToBytes(File f) throws IOException {
    int size = (int) f.length();
    byte bytes[] = new byte[size];
    byte tmpBuff[] = new byte[size];
    FileInputStream fis= new FileInputStream(f);
    try {

      int read = fis.read(bytes, 0, size);
      if (read < size) {
        int remain = size - read;
        while (remain > 0) {
          read = fis.read(tmpBuff, 0, remain);
          System.arraycopy(tmpBuff, 0, bytes, size - remain, read);
          remain -= read;
        }
      }
    }  catch (IOException e){
      throw e;
    } finally {
      fis.close();
    }

    return bytes;
  }
  private void writeInt(final DataOutputStream output, final int value) throws IOException {
    output.write(value >> 0);
    output.write(value >> 8);
    output.write(value >> 16);
    output.write(value >> 24);
  }

  private void writeShort(final DataOutputStream output, final short value) throws IOException {
    output.write(value >> 0);
    output.write(value >> 8);
  }

  private void writeString(final DataOutputStream output, final String value) throws IOException {
    for (int i = 0; i < value.length(); i++) {
      output.write(value.charAt(i));
    }
  }


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    // Set up the UI.
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_sc_activity_speech);

    startService(savedInstanceState);

    try {
      mSocket = IO.socket("http://192.168.1.104:8000");
      mSocket.connect();
      Hashtable<String, String> my_dict = new Hashtable<String, String>();
      my_dict.put("message","Hi From Phone");
      my_dict.put("handle", "phone");
      mSocket.emit("chat",my_dict);
    } catch (URISyntaxException e) {
      e.printStackTrace();
    }

    pos_counter = (Button) findViewById(R.id.pos_counter);
    neg_counter = (Button) findViewById(R.id.neg_counter);
    foreground_button = (Button) findViewById(R.id.foreground_button);
    pos_counter.setBackgroundColor(getResources().getColor(android.R.color.holo_green_light));
    neg_counter.setBackgroundColor(getResources().getColor(android.R.color.holo_red_light));

    //Set up button
    pos_counter.setOnClickListener(new View.OnClickListener(){
      @Override
      public void onClick(View v){
        pos_count++;
        pos_counter.setBackgroundColor(getResources().getColor(android.R.color.holo_green_dark));
        user_input = "0";
        handler.postDelayed(
                new Runnable() {
                  @Override
                  public void run() {
                    pos_counter.setBackgroundResource(
                            R.drawable.round_corner_text_bg_unselected);
                    pos_counter.setBackgroundColor(getResources().getColor(android.R.color.holo_green_light));
                    mic_input = "-1";
                  }
                },
                500);
      }
    });

    neg_counter.setOnClickListener(new View.OnClickListener(){
      @Override
      public void onClick(View v){
        neg_count++;
        neg_counter.setBackgroundColor(getResources().getColor(android.R.color.holo_red_dark));
        //txv_neg.setText(NEG_COUNTER_HEADER + Integer.toString(neg_count));
        user_input = "1";
        handler.postDelayed(
                new Runnable() {
                  @Override
                  public void run() {
                    neg_counter.setBackgroundResource(
                            R.drawable.round_corner_text_bg_unselected);
                    neg_counter.setBackgroundColor(getResources().getColor(android.R.color.holo_red_light));
                    mic_input = "-1";
                  }
                },
                500);
      }
    });

    foreground_button.setOnClickListener(new View.OnClickListener(){
      @Override
      public void onClick(View v) {
        stopService(savedInstanceState);
      }
    });

    String actualLabelFilename = LABEL_FILENAME.split("file:///android_asset/", -1)[1];
    Log.i(LOG_TAG, "Reading labels from: " + actualLabelFilename);
    BufferedReader br = null;
    try {
      br = new BufferedReader(new InputStreamReader(getAssets().open(actualLabelFilename)));
      String line;
      while ((line = br.readLine()) != null) {
        labels.add(line);
        if (line.charAt(0) != '_') {
          displayedLabels.add(line.substring(0, 1).toUpperCase() + line.substring(1));
        }
      }
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!", e);
    }

    recognizeCommands =
            new RecognizeCommands(
                    labels,
                    AVERAGE_WINDOW_DURATION_MS,
                    DETECTION_THRESHOLD,
                    SUPPRESSION_MS,
                    MINIMUM_COUNT,
                    MINIMUM_TIME_BETWEEN_SAMPLES_MS);

    String actualModelFilename = MODEL_FILENAME.split("file:///android_asset/", -1)[1];
    try {
      tfLite = new Interpreter(loadModelFile(getAssets(), actualModelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    tfLite.resizeInput(0, new int[] {RECORDING_LENGTH, 1});
    tfLite.resizeInput(1, new int[] {1});

    requestMicrophonePermission();
    startRecording();
    startRecognition();

    sampleRateTextView = findViewById(R.id.sample_rate);
    inferenceTimeTextView = findViewById(R.id.inference_info);
    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    gestureLayout = findViewById(R.id.gesture_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);

    threadsTextView = findViewById(R.id.threads);
    plusImageView = findViewById(R.id.plus);
    minusImageView = findViewById(R.id.minus);
    apiSwitchCompat = findViewById(R.id.api_info_switch);

    yesTextView = findViewById(R.id.yes);
    noTextView = findViewById(R.id.no);

    apiSwitchCompat.setOnCheckedChangeListener(this);

    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
            new ViewTreeObserver.OnGlobalLayoutListener() {
              @Override
              public void onGlobalLayout() {
                gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                int height = gestureLayout.getMeasuredHeight();

                sheetBehavior.setPeekHeight(height);
              }
            });
    sheetBehavior.setHideable(false);

    sheetBehavior.setBottomSheetCallback(
            new BottomSheetBehavior.BottomSheetCallback() {
              @Override
              public void onStateChanged(@NonNull View bottomSheet, int newState) {
                switch (newState) {
                  case BottomSheetBehavior.STATE_HIDDEN:
                    break;
                  case BottomSheetBehavior.STATE_EXPANDED:
                  {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                  }
                  break;
                  case BottomSheetBehavior.STATE_COLLAPSED:
                  {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                  }
                  break;
                  case BottomSheetBehavior.STATE_DRAGGING:
                    break;
                  case BottomSheetBehavior.STATE_SETTLING:
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                    break;
                }
              }

              @Override
              public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
            });

    plusImageView.setOnClickListener(this);
    minusImageView.setOnClickListener(this);

    sampleRateTextView.setText(SAMPLE_RATE + " Hz");
    Runnable sound = new Runnable() {
      public void run() {
        mSpeechRecognizer.startListening(mSpeechRecognizerIntent);
        handler.postDelayed(
                new Runnable() {
                  @Override
                  public void run() {
                    mSpeechRecognizer.stopListening();
                  }
                },
                1000);
      }
    };
  }

  private void startService(Bundle savedInstanceState) {
    Intent serviceIntent = new Intent(this, MicService.class);
    ContextCompat.startForegroundService(this,serviceIntent);
  }

  public void stopService(Bundle savedInstanceState) {
    Intent serviceIntent = new Intent(this, MicService.class);
    stopService(serviceIntent);
  }

  public ArrayList<Float> pythonTFfeatures(float[] sound) {
    Python python = Python.getInstance();
    PyObject pythonFile = python.getModule("TF_Features");
    List<PyObject> Sound =  pythonFile.callAttr("feautres", sound).asList();
    ArrayList<Float> processedSound = new ArrayList(40);
    for(PyObject listObject : Sound){
      processedSound.add(listObject.toFloat());
    }
    return processedSound;
  }

  private void requestMicrophonePermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      requestPermissions(
              new String[] {android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
    }
  }

  @Override
  public void onRequestPermissionsResult(
          int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_RECORD_AUDIO
            && grantResults.length > 0
            && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

      android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

      mSpeechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
              RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
      mSpeechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE,
              Locale.getDefault());

      mSpeechRecognizer.setRecognitionListener(new RecognitionListener() {


                                                 @Override
                                                 public void onReadyForSpeech(Bundle bundle) {

                                                 }

                                                 @Override
                                                 public void onBeginningOfSpeech() {

                                                 }

                                                 @Override
                                                 public void onRmsChanged(float v) {

                                                 }

                                                 @Override
                                                 public void onBufferReceived(byte[] bytes) {

                                                 }

                                                 @Override
                                                 public void onEndOfSpeech() {

                                                 }

                                                 @Override
                                                 public void onError(int i) {

                                                 }

                                                 @Override
                                                 public void onResults(Bundle bundle) {
                                                   ArrayList<String> matches = bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);

                                                   //check if there is a translation
                                                   if (matches != null)
                                                      isSpeech = true;

                                                 }

                                                 @Override
                                                 public void onPartialResults(Bundle bundle) {

                                                 }

                                                 @Override
                                                 public void onEvent(int i, Bundle bundle) {

                                                 }
                                               });
    }
  }

  public synchronized void startRecording() {
    if (recordingThread != null) {
      return;
    }
    shouldContinue = true;
    recordingThread =
            new Thread(
                    new Runnable() {
                      @Override
                      public void run() {
                        record();
                      }
                    });
    recordingThread.start();
  }

  public synchronized void stopRecording() {
    if (recordingThread == null) {
      return;
    }
    shouldContinue = false;
    recordingThread = null;
  }

  private byte[] short2byte(short[] sData) {
    int shortArrsize = sData.length;
    byte[] bytes = new byte[shortArrsize * 2];
    for (int i = 0; i < shortArrsize; i++) {
      bytes[i*2] = (byte)(sData[i] & 0x00FF);
      bytes[(i * 2) + 1] = (byte) (sData[i] >> 8);
      sData[i] = 0;
    }
    return bytes;
  }

  private void record() {
    if(isSpeech != true) {
      int bufferSize =
              AudioRecord.getMinBufferSize(
                      SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
      if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
        bufferSize = SAMPLE_RATE * 2;
      }
      short[] audioBuffer = new short[bufferSize / 2];

      AudioRecord record =
              new AudioRecord(
                      MediaRecorder.AudioSource.DEFAULT,
                      SAMPLE_RATE,
                      AudioFormat.CHANNEL_IN_MONO,
                      AudioFormat.ENCODING_PCM_16BIT,
                      bufferSize);

      if (record.getState() != AudioRecord.STATE_INITIALIZED) {
        Log.e(LOG_TAG, "Audio Record can't initialize!");
        return;
      }

      record.startRecording();

      Log.v(LOG_TAG, "Start recording");
      NoiseSuppressor noiseSuppressor = null;
      while (shouldContinue) {
        if(android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.JELLY_BEAN)
        {
          noiseSuppressor = NoiseSuppressor.create(record.getAudioSessionId());
        }

        int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
        int maxLength = recordingBuffer.length;
        int newRecordingOffset = recordingOffset + numberRead;
        int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
        int firstCopyLength = numberRead - secondCopyLength;
        recordingBufferLock.lock();
        try {
          System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
          System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
          recordingOffset = newRecordingOffset % maxLength;
        } finally {
          recordingBufferLock.unlock();
        }
      }

      record.stop();
      record.release();
      noiseSuppressor.release();

    }

  }

  public synchronized void startRecognition() {
    if (recognitionThread != null) {
      return;
    }
    shouldContinueRecognition = true;
    recognitionThread =
            new Thread(
                    new Runnable() {
                      @Override
                      public void run() {
                        recognize();
                      }
                    });
    recognitionThread.start();
  }

  public synchronized void stopRecognition() {
    if (recognitionThread == null) {
      return;
    }
    shouldContinueRecognition = false;
    recognitionThread = null;
  }

  private void recognize() {

    Log.v(LOG_TAG, "Start recognition");

    short[] inputBuffer = new short[RECORDING_LENGTH];
    float[][] floatInputBuffer = new float[RECORDING_LENGTH][1];
    float[][] outputScores = new float[1][labels.size()];
    int[] sampleRateList = new int[] {SAMPLE_RATE};

    // Loop, grabbing recorded data and running the recognition model on it.
    while (shouldContinueRecognition) {
      long startTime = new Date().getTime();
      recordingBufferLock.lock();
      try {
        int maxLength = recordingBuffer.length;
        int firstCopyLength = maxLength - recordingOffset;
        int secondCopyLength = recordingOffset;
        System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
        System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
        System.arraycopy(inputBuffer, 0,copyForSaving,0,recordingBuffer.length);
        Data = short2byte(copyForSaving);
      } finally {
        recordingBufferLock.unlock();
      }

      for (int i = 0; i < RECORDING_LENGTH; ++i) {
        floatInputBuffer[i][0] = inputBuffer[i] / 32767.0f;
      }

      Object[] inputArray = {floatInputBuffer, sampleRateList};
      Map<Integer, Object> outputMap = new HashMap<>();
      outputMap.put(0, outputScores);

      tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

      long currentTime = System.currentTimeMillis();
      ut_start = System.currentTimeMillis() / 1000L;
      RecognizeCommands.RecognitionResult result = recognizeCommands.processLatestResults(outputScores[0], currentTime);
      lastProcessingTimeMs = new Date().getTime() - startTime;

      runOnUiThread(
              new Runnable() {
                @Override
                public void run() {

                  inferenceTimeTextView.setText(lastProcessingTimeMs + " ms");

                  if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
                    int labelIndex = -1; //labelIndex = -1
                    selectedTextView = yesTextView;
                    for (int i = 0; i < labels.size(); ++i) {
                      if (labels.get(i).equals(result.foundCommand)) {
                        labelIndex = i;
                      }
                    }
                    switch (labelIndex - 2) {
                      case 0:
                        selectedTextView = yesTextView;
                        break;
                      default:
                        selectedTextView = noTextView;
                        break;
                    }
                    if (selectedTextView != null) {
                      ut = System.currentTimeMillis() / 1000L;
                      if(selectedTextView == yesTextView) {
                        selectedTextView.setBackgroundResource(R.drawable.round_corner_text_bg_selected);
                        selectedTextView.setBackgroundColor(getResources().getColor(android.R.color.holo_green_light));
                        selectedTextView.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
                      }
                      if(selectedTextView == noTextView) { //explicit separate if statement for no
                        selectedTextView.setBackgroundResource(R.drawable.round_corner_text_bg_selected);
                        selectedTextView.setBackgroundColor(getResources().getColor(android.R.color.holo_red_light));
                        selectedTextView.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
                      }
                      handler.postDelayed(
                              new Runnable() {
                                @Override
                                public void run() {
                                  if(selectedTextView == yesTextView) {
                                    pos++;
                                    selectedTextView.setText(POS_COUNTER + Integer.toString(pos));
                                    mic_input = "0";
                                    Hashtable<String, String> my_dict = new Hashtable<String, String>();
                                    my_dict.put("message","0");
                                    my_dict.put("handle", "phone");
                                    mSocket.emit("chat",my_dict);

                                    timeInput.add(ut);
                                    startInput.add(ut_start);
                                    userInput.add(user_input);
                                    micInput.add(mic_input);
                                    StringBuffer sb = new StringBuffer();
                                    for(int i = 0; i < timeInput.size(); i++){
                                      sb.append(timeInput.get(i));
                                      sb.append(",");
                                      sb.append(startInput.get(i));
                                      sb.append(",");
                                      sb.append(userInput.get(i));
                                      sb.append(",");
                                      sb.append(micInput.get(i));
                                      sb.append("\n");
                                    }
                                    String str = sb.toString();
                                    FileOutputStream fos = null;
                                    try{
                                      fos = openFileOutput(FILE_NAME_User, MODE_PRIVATE);
                                      fos.write(str.getBytes());
                                      Toast.makeText(SpeechActivity.this,"Saved to " + getFilesDir() + "/" + FILE_NAME_User, Toast.LENGTH_SHORT).show();
                                    }catch (FileNotFoundException e){
                                      e.printStackTrace();
                                    }catch (IOException e){
                                      e.printStackTrace();
                                    }finally {
                                      if ((fos != null)) {
                                        try{
                                          fos.close();
                                        }catch (IOException e) {
                                          e.printStackTrace();
                                        }
                                      }
                                    }

                                    handler.postDelayed(
                                            new Runnable() {
                                              @Override
                                              public void run() {
                                                pos_counter.setBackgroundResource(
                                                        R.drawable.round_corner_text_bg_unselected);
                                                pos_counter.setBackgroundColor(getResources().getColor(android.R.color.holo_green_light));
                                                mic_input = "-1";
                                              }
                                            },
                                            500);

                                    FileOutputStream os = null;
                                    try {
                                      os = new FileOutputStream(getFilesDir() + "/" + ut +"valid.pcm");
                                      os.write(Data, 0, RECORDING_LENGTH);
                                      File pcm = new File(getFilesDir() + "/" + ut +"valid.pcm");
                                      File wav = new File(getFilesDir() + "/" + ut +"valid.wav");
                                      rawToWave(pcm, wav);
                                      os.close();
                                    } catch (IOException e) {
                                      e.printStackTrace();
                                    }
                                  }
                                  if(selectedTextView == noTextView){ //explicit separate if statement for no
                                    neg++;
                                    selectedTextView.setText(NEG_COUNTER + Integer.toString(neg));
                                    mic_input = "1";
                                    Hashtable<String, String> my_dict = new Hashtable<String, String>();
                                    my_dict.put("message","1");
                                    my_dict.put("handle", "phone");
                                    mSocket.emit("chat",my_dict);

                                    timeInput.add(ut);
                                    startInput.add(ut_start);
                                    userInput.add("1");
                                    micInput.add(mic_input);
                                    StringBuffer sb = new StringBuffer();
                                    for(int i = 0; i < timeInput.size(); i++){
                                      sb.append(timeInput.get(i));
                                      sb.append(",");
                                      sb.append(startInput.get(i));
                                      sb.append(",");
                                      sb.append(userInput.get(i));
                                      sb.append(",");
                                      sb.append(micInput.get(i));
                                      sb.append("\n");

                                    }
                                    String str = sb.toString();
                                    FileOutputStream fos = null;
                                    try{
                                      fos = openFileOutput(FILE_NAME_User, MODE_PRIVATE);
                                      fos.write(str.getBytes());
                                      Toast.makeText(SpeechActivity.this,"Saved to " + getFilesDir() + "/" + FILE_NAME_User, Toast.LENGTH_SHORT).show();
                                    }catch (FileNotFoundException e){
                                      e.printStackTrace();
                                    }catch (IOException e){
                                      e.printStackTrace();
                                    }finally {
                                      if ((fos != null)) {
                                        try{
                                          fos.close();
                                        }catch (IOException e) {
                                          e.printStackTrace();
                                        }
                                      }
                                    }
                                    handler.postDelayed(
                                            new Runnable() {
                                              @Override
                                              public void run() {
                                                neg_counter.setBackgroundResource(
                                                        R.drawable.round_corner_text_bg_unselected);
                                                neg_counter.setBackgroundColor(getResources().getColor(android.R.color.holo_red_light));
                                                mic_input = "-1";
                                              }
                                            },
                                            500);

                                    FileOutputStream os = null;
                                    try {
                                      os = new FileOutputStream(getFilesDir() + "/" + ut +"invalid.pcm");
                                      os.write(Data, 0, RECORDING_LENGTH);
                                      File pcm = new File(getFilesDir() + "/" + ut +"invalid.pcm");
                                      File wav = new File(getFilesDir() + "/" + ut +"invalid.wav");
                                      rawToWave(pcm, wav);
                                      os.close();
                                    } catch (IOException e) {
                                      e.printStackTrace();
                                    }
                                  }
                                  selectedTextView.setBackgroundResource(
                                          R.drawable.round_corner_text_bg_unselected);
                                  selectedTextView.setTextColor(
                                          getResources().getColor(android.R.color.darker_gray));
                                }
                              },
                              2000);
                    }
                  }
                }
              });
      try {
        Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
      } catch (InterruptedException e) {
      }
    }

    Log.v(LOG_TAG, "End recognition");
  }



  @Override
  public void onClick(View v) {
    if (v.getId() == R.id.plus) {
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      numThreads++;
      threadsTextView.setText(String.valueOf(numThreads));
      //            tfLite.setNumThreads(numThreads);
      int finalNumThreads = numThreads;
      backgroundHandler.post(() -> tfLite.setNumThreads(finalNumThreads));
    } else if (v.getId() == R.id.minus) {
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads == 1) {
        return;
      }
      numThreads--;
      threadsTextView.setText(String.valueOf(numThreads));
      tfLite.setNumThreads(numThreads);
      int finalNumThreads = numThreads;
      backgroundHandler.post(() -> tfLite.setNumThreads(finalNumThreads));
    }
  }

  @Override
  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
    backgroundHandler.post(() -> tfLite.setUseNNAPI(isChecked));
    if (isChecked) apiSwitchCompat.setText("NNAPI");
    else apiSwitchCompat.setText("TFLITE");
  }

  private static final String HANDLE_THREAD_NAME = "CameraBackground";

  private void startBackgroundThread() {
    backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
    backgroundThread.start();
    backgroundHandler = new Handler(backgroundThread.getLooper());
  }

  private void stopBackgroundThread() {
    backgroundThread.quitSafely();
    try {
      backgroundThread.join();
      backgroundThread = null;
      backgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e("amlan", "Interrupted when stopping background thread", e);
    }
  }

  @Override
  protected void onResume() {
    super.onResume();

    startBackgroundThread();
  }

  @Override
  protected void onStop() {
    super.onStop();
    stopBackgroundThread();
  }

  @Override
  public void onDestroy() {

    super.onDestroy();
    Intent savedInstanceState;

  }


}
