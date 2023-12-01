import Image from 'next/image'

export default function Home() {
  return (
   <>

<header className='head sticky-top'>
  <div className="container-fluid mobContainer"> 
<div className=" nav d-flex align-items-center">
  <input type="checkbox" id="nav-check"/>
  <div className="nav-header">
    <div className="nav-title">
      <figure>
        <Image src="https://unsplash.it/500" alt="logo" width={50} height={50} />
      </figure>
    </div>
  </div>
  <div className="nav-btn">
    <label htmlFor="nav-check">
      <span></span>
      <span></span>
      <span></span>
    </label>
  </div>
  
  <div className="nav-links">
    <a href="/">Home</a>
    <a href="/about">About</a>
    <a href="/contact">Contact</a>
  </div>
  </div>
  </div>


</header>


   <div className="bg-light"> 


<div className="container-fluid">
<div className="sliderBg" >

<div className="row  ">
  <div className="col-lg-8 col-md-7 col-xs-12  ">
     <article>
        <h3>Hesdsdffd</h3>
        <p>Lorem ipsum dolor sit, amet consectetur adipisicing elit. Voluptatibus, distinctio!</p>
        <p>Lorem ipsum dolor sit amet consectetur adipisicing.</p>
        <button className=' btn btn-outline-success'>
           <a href="#" className='text-black text-decoration-none'>Explore More</a>
        </button>
      
     </article>
  </div>

  <div className="col-lg-4 col-md-5 col-xs-12 ">
  <div id="myCarousel" className="mt-2 carousel slide" data-ride="carousel">
<div className="carousel-inner">
<div className="carousel-item active">
  <Image src="https://unsplash.it/900" className="d-block w-100" alt="Slide 1" width={500} height={500}/>
  <div className="carousel-caption d-none d-md-block">
    <h5>Third slide label 1</h5>
    <p>Some representative placeholder content for the third slide.</p>
  </div>
</div>
<div className="carousel-item">
  <Image src="https://unsplash.it/1200" className="d-block w-100" alt="Slide 2" width={500} height={500}/>
  <div className="carousel-caption d-none d-md-block">
    <h5>Third slide label 2</h5>
    <p>Some representative placeholder content for the third slide.</p>
  </div>
</div>
<div className="carousel-item">
  <Image src="https://unsplash.it/1500" className="d-block w-100" alt="Slide 3" width={500} height={500} />
  <div className="carousel-caption d-none d-md-block">
    <h5>Third slide label</h5>
    <p>Some representative placeholder content for the third slide.</p>
  </div>
</div>
</div>
<a className="carousel-control-prev" href="#myCarousel" role="button" data-slide="prev">
<span className="carousel-control-prev-icon" aria-hidden="true"></span>
<span className="sr-only">Previous</span>
</a>
<a className="carousel-control-next" href="#myCarousel" role="button" data-slide="next">
<span className="carousel-control-next-icon" aria-hidden="true"></span>
<span className="sr-only">Next</span>
</a>
</div>
  </div>
</div>


</div>
</div>
</div>
<br></br>

<div className="container-fluid">
  <div className='row '>
     <div className="col-xs-12 col-md-6 col-lg-6">
        <div className="col-12 w-100 h-100">
           <article className='article w-100'>
              <h3>sdsfsdfsdfsdfsdfsdfsd</h3>
              <h1>Lorem ipsum dolor sit amet.</h1>
              <h4>sdsfsdfsdfdsdf</h4>
              <p>
                 Lorem ipsum dolor sit amet consectetur adipisicing elit. Commodi nulla vero, cum assumenda quae cumque laudantium quos omnis dolor. Mollitia eum repellat error autem asperiores aperiam vitae? Rerum officiis doloribus dolores, molestiae ullam iusto.
              </p>
            
              <button className='btn btn-outline-success'>
              Read More Blogs
              </button>
           </article>
        </div>
     </div>
     <div className="col-xs-12 col-md-6 col-lg-6">
        <div className="text-right col-12 h-100">
           <figure>
              <Image src="https://unsplash.it/1900" alt="" className='blogImage rounded w-100 h-100 '  width={500} height={500}  />
           </figure>
        </div>
     </div>
  </div>


<br></br>
<br></br>
  {/* bLOGS sECTION STARTS VISIBLE */}

  <article className='blogStories'> 
  <div className="articleStories row" >

    <div className="col-12 p-1">
    <h1 className='text-center'> Browse Popular Blogs </h1>
    <br></br>
    <br></br>
    </div>

   
     <div className="col-sm-6 col-xs-12 col-lg-3">
        <div className="card" >
           <Image className="card-Image-top" src="https://unsplash.it/1700" alt="Card Image cap" width={500} height={500}/>
           <div className="card-body">
              <h5 className="card-title">Card title</h5>
              <p className="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
              <a href="#" className ="btn btn-outline-primary">Go somewhere</a>
           </div>
        </div>
     </div>
     <div className="col-sm-6 col-xs-12 col-lg-3">
        <div className="card" >
           <Image className="card-Image-top" src="https://unsplash.it/1701" alt="Card Image cap"  width={500} height={500} />
           <div className="card-body">
              <h5 className="card-title">Card title</h5>
              <p className="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
              <a href="#" className ="btn btn-outline-primary">Go somewhere</a>
           </div>
        </div>
     </div>
     <div className="col-sm-6 col-xs-12 col-lg-3">
        <div className="card" >
           <Image className="card-Image-top" src="https://unsplash.it/1702" alt="Card Image cap"  width={500} height={500} />
           <div className="card-body">
              <h5 className="card-title">Card title</h5>
              <p className="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
              <a href="#" className ="btn btn-outline-primary">Go somewhere</a>
           </div>
        </div>
     </div>
     <div className="col-sm-6 col-xs-12 col-lg-3">
        <div className="card" >
           <Image className="card-Image-top" src="https://unsplash.it/1705" alt="Card Image cap"  width={500} height={500} />
           <div className="card-body">
              <h5 className="card-title">Card title</h5>
              <p className="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
             <a href="#" className ="btn btn-outline-primary">Go somewhere</a>
           </div>
        </div>
     </div>



 
  </div>
  {/* SHOW AND HIDE BUTTON */}
  <div>

</div>
{/* blogs section visible ends */}



  {/* Blogs Section hidden */}
<div className="collapse" id="demo"> 
<br></br>
<br></br>
  <div className="row " >
     <div className="col-sm-6 col-xs-12 col-lg-3">
        <div className="card" >
           <Image className="card-Image-top" src="https://unsplash.it/1700" alt="Card Image cap"  width={500} height={500} />
           <div className="card-body">
              <h5 className="card-title">Card title</h5>
              <p className="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
              <a href="#" className ="btn btn-outline-primary">Go somewhere</a>
           </div>
        </div>
     </div>
     <div className="col-sm-6 col-xs-12 col-lg-3">
        <div className="card" >
           <Image className="card-Image-top" src="https://unsplash.it/1701" alt="Card Image cap"  width={500} height={500} />
           <div className="card-body">
              <h5 className="card-title">Card title</h5>
              <p className="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
              <a href="#" className ="btn btn-outline-primary">Go somewhere</a>
           </div>
        </div>
     </div>
     <div className="col-sm-6 col-xs-12 col-lg-3">
        <div className="card" >
           <Image className="card-Image-top" src="https://unsplash.it/1702" alt="Card Image cap"  width={500} height={500}  />
           <div className="card-body">
              <h5 className="card-title">Card title</h5>
              <p className="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
              <a href="#" className ="btn btn-outline-primary">Go somewhere</a>
           </div>
        </div>
     </div>
     <div className="col-sm-6 col-xs-12 col-lg-3">
        <div className="card" >
           <Image className="card-Image-top" src="https://unsplash.it/1705" alt="Card Image cap" width={500} height={500}  />
           <div className="card-body">
              <h5 className="card-title">Card title</h5>
              <p className="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
              <a href="#" className ="btn btn-outline-primary">Go somewhere</a>
           </div>
        </div>
     </div>
  </div>


 

  {/* SHOW AND HIDE BUTTON */}
  <div>

</div>
</div>
<br></br>
<div className="text-center">
<a href="#demo" className="btn btn-outline-info p-2" style={{ minWidth: "10rem"}} data-toggle="collapse">Show/Hide</a>
</div>

</article>

<br></br>


<div className="container-fluid mt-5">
<h2 className="text-center mb-4">Multiple Column Slider </h2>
<br></br>
<div id="multiColumnSlider" className="carousel slide" data-ride="carousel">
<div className="carousel-inner">
  <div className="carousel-item active">
    <div className="row">
      <div className="col-3  col-sm-3 col-md-3">
        <Image src="https://placekitten.com/300/200" className="d-block w-100" alt="Slide 1" width={300} height={300}/>
      </div>
      <div className=" col-3  col-sm-3  col-md-3">
        <Image src="https://placekitten.com/300/201" className="d-block w-100" alt="Slide 2" width={300} height={300}/>
      </div>
      <div className=" col-3  col-sm-3  col-md-3">
        <Image src="https://placekitten.com/300/202" className="d-block w-100" alt="Slide 3" width={300} height={300}/>
      </div>
      <div className=" col-3  col-sm-3  col-md-3">
        <Image src="https://placekitten.com/300/206" className="d-block w-100" alt="Slide 3" width={300} height={300}/>
      </div>

    </div>
  </div>

  <div className="carousel-item">
    <div className="row">
    <div className="col-3  col-sm-3 col-md-3">
        <Image src="https://placekitten.com/300/207" className="d-block w-100" alt="Slide 1" width={300} height={300}/>
      </div>
      <div className=" col-3  col-sm-3 col-md-3">
        <Image src="https://placekitten.com/300/211" className="d-block w-100" alt="Slide 2" width={300} height={300}/>
      </div>
      <div className="col-3  col-sm-3 col-md-3">
        <Image src="https://placekitten.com/300/203" className="d-block w-100" alt="Slide 3" width={300} height={300}/>
      </div>
      <div className="col-3 col-sm-3 col-md-3">
        <Image src="https://placekitten.com/300/204" className="d-block w-100" alt="Slide 3" width={300} height={300}/>
      </div>
    </div>
  </div>
</div>

<a className="carousel-control-prev" href="#multiColumnSlider" role="button" data-slide="prev">
  <span className="carousel-control-prev-icon" aria-hidden="true"></span>
  <span className="sr-only">Previous</span>
</a>
<a className="carousel-control-next" href="#multiColumnSlider" role="button" data-slide="next">
  <span className="carousel-control-next-icon" aria-hidden="true"></span>
  <span className="sr-only">Next</span>
</a>
</div>
</div>


</div>





<br></br>

<br></br>

<hr></hr>
<div className="container-fluid">
  <div className='row d-flex align-items-center'>

     <div className="col-xs-12 col-md-6 col-lg-6">
        <div className=" col-12 w-100 h-100">
           <figure>
              <Image src="https://unsplash.it/900" alt="" className='blogImage h-100 rounded' style={{ width: "100%", maxWidth: "100%"}} width={500} height={500}/>
           </figure>
        </div>
     </div>

     <div className="col-xs-12 col-md-6 col-lg-6">
        <div className="col-12 w-100 h-100">
           <article className='article w-100'>
              <h3>sdsfsdfsdfsdfsdfsdfsd</h3>
              <h1>Lorem ipsum dolor sit amet.</h1>
              <h4>sdsfsdfsdfdsdf</h4>
              <p>
                 Lorem ipsum dolor sit amet consectetur adipisicing elit. Commodi nulla vero, cum assumenda quae cumque laudantium quos omnis dolor. Mollitia eum repellat error autem asperiores aperiam vitae? Rerum officiis doloribus dolores, molestiae ullam iusto.
              </p>
            
              <button className='btn btn-outline-success'>
              Read More Blogs
              </button>
           </article>
        </div>
     </div>

  </div>
</div>



<footer className='mt-3'>
<div className="container-fluid">
  <div className="row d-flex align-items-center p-2">
    <div className="col-6">
      <div className="d-flex align-items-center">
      <figure>
        <Image src="https://unsplash.it/100" alt="logo" className='w-50 h-50' width={50} height={50}/>
      </figure>
      <span>
      © 2021 Company, Inc
      </span>
   
    </div>
      </div>
  

    <div className="col-6">
        <div className="d-flex justify-content-end"> 
      <nav >
        <ul className='p-auto m-auto'>
          <li>
            <a href="#"><i className='fab fa-facebook'></i></a>
                    {/* Social Media Icons */}
       
          </li>
          <li><a href="#"><i className='fab fa-instagram'> </i></a></li>
          <li><a href="#"><i className='fab fa-youtube'>  </i></a></li>
        </ul>
      </nav>
      </div>
    </div>
  </div>
  </div>
</footer>

   </>
  )
}
