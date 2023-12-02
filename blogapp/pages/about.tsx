import Header from '../app/components/header';
import Footer from '../app/components/footer';
import 'bootstrap/dist/css/bootstrap.min.css';
export default function About() {
return (
<div>
   <Header/>
   <div className='container-fluid'>
      <div className='row d-flex align-items-center'>
         <div className='col-12 col-sm-6'>
            <figure>
               <img src='https://unsplash.it/500' alt='' className='w-100 h-100'/>
            </figure>
         </div>
         <div className='col-12 col-sm-6'>
            <article>
               <h2>Lorem ipsum dolor sit ametsdf.</h2>
               <h1>Lorem ipsum dolor, sit sdfsdf.</h1>
               <h5> Lorem ipsum dolor sit. Lorem ipsum dolor sit amet. </h5>
            </article>
         </div>
      </div>
      <hr>
      </hr>
      <div className='row'>
         <div className='col-12' >
            <article className='text-center'>
               <h1> Trusted Website Builder </h1>
               <br></br>
               <h1 className='text-center' >
                  Everything we do, we do with our customers in mind. 
                  <br></br>
                  Your success is our priority.
               </h1>
            </article>
         </div>
      </div>
      <br></br>
      <br></br>   
      <div className='container'>
         <div className=' row'>
            <div className='col-xs-12 col-sm-4 p-1'>
               <div className='card border-0' style= {{ }}>
                  <div className='card-body shadow-lg p-4'>
                     <h5 className='card-title'>Card title</h5>
                     <h1>8,765</h1>
                     <p className='card-text'>Some quick example text to build on the card title and make up the bulk of the ards content.</p>
                  </div>
               </div>
            </div>
            <div className='col-xs-12 col-sm-4 p-1'>
               <div className='card border-0' style= {{ }}>
                  <div className='card-body shadow-lg p-4'>
                     <h5 className='card-title'>Card title</h5>
                     <h1>6,765</h1>
                     <p className='card-text'>Some quick example text to build on the card title and make up the bulk of the ards content.</p>
                  </div>
               </div>
            </div>
            <div className='col-xs-12 col-sm-4 p-1'>
               <div className='card border-0' style= {{ }}>
                  <div className='card-body shadow-lg p-4'>
                     <h5 className='card-title'>Card title</h5>
                     <h1>87,765</h1>
                     <p className='card-text'>Some quick example text to build on the card title and make up the bulk of the card content.</p>
                  </div>
               </div>
            </div>
            <br></br>  
         </div>
         <br></br>
         <hr>
         </hr>
         <br></br>
         <div className='row'>
            <div className="col-12">
               <h1 className='text-center'>
                  Top Featured in
               </h1>
            </div>
         </div>
         <br></br>
         <br></br>
         <div className='row'>
            <div className='col-3'>
              
                  <div className='card' >
                     <img className='card-img-top h-100' src='https://unsplash.it/1300' alt='Card image cap' style={{ maxWidth: '100%' }}/>
                 
               </div>
            </div>
            <div className='col-3'>
              
                  <div className='card' >
                     <img className='card-img-top h-100' src='https://unsplash.it/1200' alt='Card image cap' style={{ maxWidth: '100%' }}/>
                  </div>
             
            </div>
            <div className='col-3'>
              
                  <div className='card' >
                     <img className='card-img-top h-100' src='https://unsplash.it/900' alt='Card image cap' style={{ maxWidth: '100%' }}/>
                  </div>
               
            </div>
            <div className='col-3'>
             
                  <div className='card' >
                     <img className='card-img-top' src='https://unsplash.it/1200' alt='Card image cap h-100'  style={{ maxWidth: '100%' }}/>
                  </div>
               </div>
          
         </div>
      </div>
      <br></br>
      <hr>
      </hr>
      <div className='row d-flex align-items-center'>
         <div className='col-12 col-sm-6'>
            <div className=' h-100'>
               <figure>
                  <img src='https://unsplash.it/1700' alt='' />
               </figure>
            </div>
         </div>
         <div className='col-12 col-sm-6'>
           
               <article>
                  <h1>
                     Our story
                  </h1>
                  <h2>Lorem ipsum dolor sit amet consectetur.</h2>
                  <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Quos, consectetur.</p>
               </article>
          
         </div>
         <div className='col-12 col-sm-6'>
        
               <article>
                  <h1>
                     Our Philosophy
                  </h1>
                  <h2>Lorem ipsum dolor sit amet consectetur.</h2>
                  <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Quos, consectetur.</p>
               </article>
           
         </div>
         <div className='col-12 col-sm-6'>
            
               <figure>
                  <img src='https://unsplash.it/700' alt='' />
               </figure>
          
         </div>
      </div>
   </div>
   <Footer/>
</div>
)}