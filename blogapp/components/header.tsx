// import "./customCSS/custom.css"; 

import React from 'react';
import "./custom.css";
import 'bootstrap/dist/css/bootstrap.min.css';

export default function Header() {
    return (
<header className='head sticky-top'>
  <div className="container-fluid mobContainer"> 
<div className=" nav d-flex align-items-center">
  <input type="checkbox" id="nav-check"/>
  <div className="nav-header">
    <div className="nav-title">
      <figure>
        <img src="https://unsplash.it/500" alt="logo" style={{ width: "50px", height: "50px "}}/>
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

    )}