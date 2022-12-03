import React, {useState, useEffect} from 'react';
import {StatusBar} from 'expo-status-bar';
import {Button, CheckBox, StyleSheet, Text, View} from 'react-native';
import Slider from '@react-native-community/slider';
import {useDropzone} from 'react-dropzone';
import {Icon} from '@rneui/themed';
import "@fontsource/roboto-mono";


const thumbsContainer = {
  display: 'flex',
  flexDirection: 'row',
  flexWrap: 'wrap',
  marginTop: 16
};

const thumb = {
  display: 'inline-flex',
  borderRadius: 2,
  border: '1px solid #eaeaea',
  marginBottom: 8,
  marginRight: 8,
  width: 100,
  height: 100,
  padding: 4,
  boxSizing: 'border-box'
};

const thumbInner = {
  display: 'flex',
  minWidth: 0,
  overflow: 'hidden'
};

const img = {
  display: 'block',
  width: 'auto',
  height: '100%'
};

var contentImg, styleImg;


function Basic(props) {
  const [files, setFiles] = useState([]);
  const {getRootProps, getInputProps} = useDropzone({
    accept: {
      'image/*': []
    },
    onDrop: acceptedFiles => {
      setFiles(acceptedFiles.map(file => Object.assign(file, {
        preview: URL.createObjectURL(file)
      })));
    }
  });

  const thumbs = files.map(file => (
    <div style={thumb} key={file.name}>
      <div style={thumbInner}>
        <img
          src={file.preview}
          style={img}
          // Revoke data uri after image is loaded
          onLoad={() => { URL.revokeObjectURL(file.preview) }}
        />
      </div>
    </div>
  ));

  useEffect(() => {
    // Make sure to revoke the data uris to avoid memory leaks, will run on unmount
    return () => files.forEach(file => URL.revokeObjectURL(file.preview));
  }, []);

  return (
    <section className="container">
      <div {...getRootProps({className: 'dropzone'})}>
        <input {...getInputProps()} />
        <View style={{flex: 1, border: "1px solid white", textAlign: "center", alignSelf: "center", justifyContent: "center", padding: 10, borderRadius: 10}}>
          <div {...getRootProps({className: 'dropzone'})}>
            <input {...getInputProps()} />
            <Text style={{fontWeight: 500, color: "white", fontFamily: "Roboto Mono"}}>Upload {props.name} image</Text>
          </div>
        </View>
      </div>
      <aside style={thumbsContainer}>
        {thumbs}
      </aside>
    </section>
  );
}


const MainContainer = () => {
  return (
    <View style={styles.container}>
        <>
        <Header/>
        <View style={{flex: 0.7, padding: 20, paddingRight: 0, backgroundColor: "whitesmoke", borderRadius: 10, width: 1200}}>
          <>
          <Sandbox/>
          </>
        </View>
        <Footer/>
        </>
    </View>
  );
};

/**
Header element at the top of the page.
*/
const Header = () => {
    return (
        <View style={styles.headerContainer}>
            <>
            <Text style={styles.header}>
                Neural Style Transfer Playground (Gatys et al. method) 🎨 🖼️
            </Text>
            <Text style={styles.subHeader}>
                <i>Style transfer</i> is the process of generating an image that
                retains the content of one image and shares the style of another
                image.
            </Text>
            </>
        </View>
    );
};

/**
Footer element at the bottom of the page.
*/
const Footer = () => {
    return (
        <View style={styles.footer}>
            <Text style={{fontSize: 16, fontFamily: "Roboto Mono"}}>
                Implementation by&nbsp;
                <a href="https://github.com/kianzohoury/style_transfer">
                    @kianzohoury
                </a>
            </Text>
        </View>
    );
};


/**
Slider element for each adjustable hyperparameter.
*/
const SliderElement = (props) => {
    const [currVal, setVal] = useState(props.default);
    const [isHovering, setHovering] = useState(false);
    const handleOnMouseOver = () => {
        setHovering(true);
    };
    const handleOnMouseOut = () => {
        setHovering(false);
    };

    return (
        <View style={styles.sliderContainer}>
            <Text
                style={styles.sliderLabel}
                onMouseOver={handleOnMouseOver}
                onMouseOut={handleOnMouseOut}>
                {props.name}
            </Text>
            <View style={styles.sliderInner}>
                <>
                <Slider
                    style={{flex: 0.8}}
                    animateTransitions
                    value={props.default}
                    minimumValue={props.min}
                    maximumValue={props.max}
                    minimumTrackTintColor="#dcdcdc"
                    maximumTrackTintColor="#3a3a3a"
                    thumbTintColor="#3a3a3a"
                    onValueChange={value => setVal(value)}
                    step={props.step}
                    snapped={true}
                />
                <Text style={{flex: 0.15, color: "#3a3a3a", fontFamily: "Roboto Mono"}}>{currVal}</Text>
                </>
            </View>
            {isHovering && props.description &&
                <Text style={styles.sliderHover}>{props.description}</Text>}
        </View>
    );
};

/**
Checkbox element for speciying image options.
*/
const CheckBoxElement = (props) => {
    const [isSelected, setSelection] = useState(props.default);
    return (
        <View style={styles.checkBoxContainer}>
            <>
            <Text style={styles.checkBoxLabel}>{props.name}</Text>
            <CheckBox
                value={isSelected}
                onValueChange={setSelection}
                color={"#3a3a3a"}
            />
            </>
        </View>
    );
};

const ButtonElement = (props) => {
    const [isDisabled, setButton] = useState(true);
    const buttonStatusHandler = () => {
        setButton(contentImg && styleImg);
    }
    return (
        <View style={{width: "50%", justifyContent: "center", alignSelf: "center", borderRadius: 5, overflow: 'hidden', backgroundColor: "#f194ff"}}>
            <Button
                disabled={isDisabled}
                title="Generate"
                color="transparent">
            </Button>
        </View>
    );
};




const Sandbox = () => {
    return (
        <View style={{
            flex: 1,
            flexDirection: "row",
            justifyContent: "space-between"
        }}
        >
            <>
            <View style={styles.leftContainer}>
                <>
                <View style={[styles.subContainer, {marginBottom: 20}]}>
                    <Text style={{color: "grey", fontWeight: "300", paddingBottom: 20, fontFamily: "Roboto Mono"}}>Hyperparameters</Text>
                    <View>
                        <View>
                            <SliderElement name="Alpha" description="Weight of the content loss." default={1.0} min={0.001} max={2.0} step={0.001}/>
                        </View>
                        <View>
                            <SliderElement name="Beta" description="Weight of the style loss." default={100000} min={0} max={1000000} step={10000}/>
                        </View>
                        <View>
                            <SliderElement name="TV Regularization" description="Strength of image smoothness." default={0.00001} min={0.000001} max={1} step={0.000001}/>
                        </View>
                        <View>
                            <SliderElement name="Learning Rate" description="Learning rate for LBFGS optimizer." default={1.0} min={0.001} max={2.0} step={0.001}/>
                        </View>
                        <View>
                            <SliderElement name="Iterations" description="Number of iterations to run." default={100} min={10} max={500} step={10}/>
                        </View>
                    </View>
                </View>
                <View style={[styles.subContainer, {marginBottom: 0, flex: 1}]}>
                    <>
                    <Text style={{color: "grey", fontWeight: "300", fontFamily: "Roboto Mono"}}>Image Options</Text>
                    <View style={{flex: 1, justifyContent: "center"}}>
                        <CheckBoxElement name="Normalize Input" default={true}/>
                        <CheckBoxElement name="Random Initialization" default={true}/>
                        <CheckBoxElement name="Luminance Transfer" default={false}/>
                        <CheckBoxElement name="Facial Preservation" default={false}/>
                    </View>
                    </>
                </View>
                </>
            </View>
            <View style={styles.rightContainer}>
                <View style={[styles.subContainer, {backgroundColor: "#3a3a3a", flex: 1, justifyContent: "space-between"}]}>
                    <>
                    <View style={{flex: 0.5}}>
                        <Basic name="content"/>
                    </View>
                    <View style={{flex: 0.5}}>
                        <Basic name="style"/>
                    </View>
                    <ButtonElement/>
                    </>
                </View>
            </View>
            </>
        </View>
    );
};




export default function App() {
  return (
    <View style={styles.container}>
      <MainContainer/>
    </View>
  );
}



const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: "column",
    alignItems: 'center'
  },
  header: {
      width: "100%",
      fontSize: 32,
      fontWeight: "bold",
      textAlign: "center",
      fontFamily: "Roboto Mono"
  },
  headerContainer: {
      position: "relative",
      width: "100%",
      flex: 0.2,
      justifyContent: "center",
      alignSelf: "center",
  },
  subHeader: {
      width: "75%",
      fontSize: 20,
      textAlign: "center",
      paddingTop: 10,
      color: "grey",
      fontFamily: "Roboto Mono",
      justifyContent: "center",
      alignSelf: "center"

  },
  footer: {
      flex: 0.1,
      width: "100%",
      justifyContent: "center",
      alignItems: "center",
  },
  sliderContainer: {
      height: 200,
      flex: 1,
      flexDirection: "column"
  },
  sliderLabel: {
      paddingBottom: 10,
      fontFamily: "Roboto Mono"
  },
  sliderHover: {
      position: "absolute",
      backgroundColor: "#3a3a3a",
      color: "white",
      padding: 10,
      borderRadius: 10,
  },
  sliderInner: {
      flex: 1,
      padding: 5,
      justifyContent: "space-between",
      alignItems: "center",
      flexDirection: "row"
  },
  checkBoxContainer: {
      flexDirection: "row",
      paddingBottom: 15,
      justifyContent: "space-between"
  },
  checkBoxLabel: {
      fontFamily: "Roboto Mono",
  },
  leftContainer: {
      flex: 0.4,
      flexDirection: "column",
      alignSelf: "center",
      height: "100%"
  },
  rightContainer: {
      flex: 0.6,
      flexDirection: "column",
      alignSelf: "center",
      height: "100%",
      paddingLeft: 20,
      paddingRight: 20
  },
  subContainer: {
      backgroundColor: "white",
      borderRadius: 10,
      padding: 20,
      boxShadow: "rgb(8 15 41 / 8%) 0.5rem 0.5rem 2rem 0px, rgb(8 15 41 / 8%) 0px 0px 1px 0px",
  },
});
