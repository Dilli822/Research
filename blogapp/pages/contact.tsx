
import Home from "../app/page";
import Footer from "../app/components/footer";
import Header from "../app/components/header";
import 'bootstrap/dist/css/bootstrap.min.css';

export default function contact() {
    return (
        <div>

            <Header/>

            <div className="contactBg" style={{ height: "500px", position: "relative"}}>
            <div className="" style={{
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  padding: "20px",
  borderRadius: "5px",
  color: "#fff",
}}>


    <h1>Contact </h1>
  {/* Your content goes here */}
</div>
                
            </div>
         
            <div className="container-fluid ">
           
            <br />

            <div className="row">
                <div className="col-12">
                <h1 className="text-center p-1 ">Let's get connected</h1>
              <br />
              <br></br>
                </div>
            </div>
              
           
            <div className="bg-light "> 
              <div className="row">
                <div className="col-xs-12  col-sm-6  col-md-6 col-lg-6 p-3">
                   <h5>Company Name Inc </h5>
                   <h6> Call us: 9812321342342 </h6>
                   <h6> Address: sdfsdfsdfs </h6>
                   <p>
                    Lorem ipsum dolor sit amet consectetur adipisicing elit. Aspernatur, magnam.
                   </p>
                   <p>
                    Lorem ipsum, dolor sit amet consectetur adipisicing elit. Reiciendis fugit, iste cumque repudiandae corrupti quos incidunt necessitatibus a deleniti aut repellendus totam alias! Accusamus, placeat.
                   </p>
                </div>

                <div className="col-xs-12 col-sm-6  col-md-6 col-lg-6">
                    <div className="col-xs-12 col-sm-12 col-lg-9 p-3 ">
                    <form action="" className="contactForm">

                    <div className="form-group">
    <label htmlFor="exampleInputFirstName">First Name</label>
    <input type="text" className="form-control" id="exampleInputEmail1" aria-describedby="FirstNameHelp" placeholder="Enter First Name" required/>
    {/* <small id="dataHelp" className="form-text text-muted">We'll never share your data with anyone else.</small> */}
  </div>


  <div className="form-group">
    <label htmlFor="exampleInputFirstName"> Last Name</label>
    <input type="text" className="form-control" id="exampleInputEmail1" aria-describedby="FirstNameHelp" placeholder="Enter Last Name" required/>
    {/* <small id="dataHelp" className="form-text text-muted">We'll never share your data with anyone else.</small> */}
  </div>


  <div className="form-group">
    <label htmlFor="exampleInputEmail1">Address</label>
    <input type="text" className="form-control" id="exampleInputEmail1" aria-describedby="addressHelp" placeholder="Enter Address" required/>
    {/* <small id="emailHelp" className="form-text text-muted">We'll never share your email with anyone else.</small> */}
  </div>



                    <div className="form-group">
    <label htmlFor="exampleInputEmail1">Email address</label>
    <input type="email" className="form-control" id="exampleInputEmail1" aria-describedby="emailHelp" placeholder="Enter email" required/>
    <small id="emailHelp" className="form-text text-muted">We'll never share your email with anyone else.</small>
  </div>

  <div className="form-group">
    <label htmlFor="exampleFormControlTextarea1">Example textarea</label>
    <textarea className="form-control" id="exampleFormControlTextarea1" rows={3}></textarea>
  </div>


  {/* <div className="form-check">
    <input type="checkbox" className="form-check-input" id="exampleCheck1" required/>
    <label className="form-check-label" htmlFor="exampleCheck1">Check me out</label>
  </div> */}
  <button type="submit" className="btn btn-outline-success " style={{ width: "10rem"}}>
   
    Submit </button>
                    </form>
                    </div>
           
                </div>
              </div>
              
            </div>
            </div>
            <br></br>
            <br></br>
            <Footer/>
        </div>
    )}
