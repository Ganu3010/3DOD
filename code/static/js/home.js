// document.getElementById("first").addEventListener("click", async function () {
//   try {
//     // Make an asynchronous request to the Flask route
//     const response = await fetch("/upload", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       // You can include data in the request body if needed
//       // body: JSON.stringify({ key: 'value' })
//     });

//     if (response.ok) {
//       const result = await response.json();
//       console.log(result.message);
//       // Perform any client-side actions based on the response
//     } else {
//       console.error("Action failed");
//     }
//   } catch (error) {
//     console.error("Error during action:", error);
//   }
// });

// const handleUpload = async () => {
//   console.log("hi");
// const formData = new FormData();
// formData.append("file", selectedFile);

//   try {
//     const response = await axios.post(
//       "http://localhost:5000/upload",
//       formData,
//       {
//         headers: {
//           "Content-Type": "multipart/form-data",
//         },
//       }
//     );

//     console.log(response.data);
//   } catch (error) {
//     console.error("Error uploading file: ", error);
//   }
// };

// const predict = async () => {
//   const formData = new FormData();
//   formData.append("file", selectedFile);

//   try {
//     const response = await axios.post(
//       "http://localhost:5000/predict",
//       formData,
//       {
//         headers: {
//           "Content-Type": "multipart/form-data",
//         },
//       }
//     );

//     console.log(response.data);
//   } catch (error) {
//     console.error("Error uploading file: ", error);
//   }
// };
