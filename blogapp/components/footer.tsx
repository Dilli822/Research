
import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTwitter, faInstagram, faFacebook } from '@fortawesome/free-brands-svg-icons';
export default function Footer() {
    return (
<footer className='mt-3'>
<div className="container-fluid">
  <div className="row d-flex align-items-center p-2">
    <div className="col-6">
      <div className="d-flex align-items-center">
      <figure>
        <img src="https://unsplash.it/100" alt="logo" className='w-50 h-50'/>
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

    )
}